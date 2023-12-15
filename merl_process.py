import numpy as np   
import scipy

import os
import sys

import merl

import util

from tqdm import tqdm
import copy

directory_path = "/home/yuan/school/cp/final_project/BRDFDatabase/brdfs/"

class MERL_Collection:
    
    materials = []
    

    #should be 300 * 1458000 in dimension
    BRDF_array : np.ndarray
    X : np.ndarray
    cos_precomputed : np.ndarray
    mean : np.ndarray

    ## valid mask
    valid_mask_precomputed : np.ndarray
    valid_col_idx : np.ndarray
    valid_offset : np.ndarray # offset used to map valid array to full array (110xxx -> 145xxx). It has a length of 145xxx, with invalid locations set to np.nan
    valid_offset_noNan : np.ndarray

    scaled_pc : np.ndarray

    epsilon = 0.001

    half_diff_meshgrid : None

    p_d : list
    theta_d : list
    theta_h : list


    size: int
    valid_size: int

    
    def __init__(self) -> None:

        if os.path.exists('./arrays/matrix.npy'):
            print("Reading BRDF matrix")
            self.BRDF_array = np.load('./arrays/matrix.npy')
            self.size = self.BRDF_array.shape[1]
        else:
            with open("./name_list") as f:
                lines = f.readlines()
                assert (len(lines) == 100)

                # development setup
                # lines = lines[:10]

                for line in lines:
                    print("Reading material " + str(line))

                    # there is an extra \n
                    if line[-1] == '\n':
                        line = directory_path + line[:-1]
                    else:
                        line = directory_path + line

                    mat = merl.MERL_BRDF(line)
                    self.materials.append(mat)
                    self.size = mat.m_size

            print("Constructing BRDF matrix")
            self.BRDF_array = np.zeros((3 * len(lines),self.size),dtype=np.float32)

            for i in tqdm(range(len(self.materials))):
                mat = self.materials[i]
                self.BRDF_array[i*3,:] = np.array(mat.r_channel_unscaled) * mat.r_scale
                self.BRDF_array[i*3+1,:] = np.array(mat.g_channel_unscaled) * mat.g_scale
                self.BRDF_array[i*3+2,:] = np.array(mat.b_channel_unscaled) * mat.b_scale

            self.BRDF_array = self.BRDF_array.astype(np.float32)
            np.save('./arrays/matrix.npy',self.BRDF_array)

        self.X = np.zeros_like(self.BRDF_array)


        #create meshgrid for half diff space
        phi_d_ = np.linspace(0,179,180, endpoint=True).astype(np.uint8)
        theta_d_ = np.linspace(0,89,90, endpoint=True).astype(np.uint8)
        theta_h_ = np.linspace(0,89,90, endpoint=True).astype(np.uint8)

        self.p_d, self.theta_d, self.theta_h = np.meshgrid(phi_d_,theta_d_,theta_h_, indexing='ij')


    # Phi got messed up but theta is correct. Let's just use the theta from util.getwdwh/util.getwiwo
    def conversion_test(self):
        rng = np.random.default_rng()
        u,v = rng.random(2)
        wi = util.random_uniform_hemisphere(u,v)
        theta_i,phi_i = util.to_spherical(wi)
        u,v = rng.random(2)
        wo = util.random_uniform_hemisphere(u,v)
        theta_o,phi_o = util.to_spherical(wo)

        result = merl.convert_to_hd(theta_i,phi_i,theta_o,phi_o)
        theta_h, phi_h, theta_d, phi_d = result[0],result[1],result[2],result[3]
        wh = util.to_cartesian(theta_h,phi_h)
        wd = util.to_cartesian(theta_d,phi_d)

        wi_t , wo_t = util.getwiwo(wh,wd)

        print("Done")

    def idx_test(self):

        for i in range(10000):
            rng = np.random.default_rng()
            u, v = rng.random(2)
            wi = util.random_uniform_hemisphere(u, v)
            theta_i, phi_i = util.to_spherical(wi)
            u, v = rng.random(2)
            wo = util.random_uniform_hemisphere(u, v)
            theta_o, phi_o = util.to_spherical(wo)
            result = merl.convert_to_hd(theta_i,phi_i,theta_o,phi_o)
            theta_h, phi_h, theta_d, phi_d = result[0],result[1],result[2],result[3]

            idx = merl.get_index_from_hall_diff_coords(theta_h,theta_d,phi_d)
            result = merl.get_half_diff_idxes_from_index(idx)

            print(result)
            print("Done")

        print("Done")

    # Find the median of each direction
    def get_reference(self):
        if os.path.exists("./arrays/X.npy"):
            print("Reading observation matrix X")
            self.X = np.load("./arrays/X.npy")
        else:
            print("Constructing observation matrix X")
            median = np.median(self.BRDF_array,axis=0).astype(np.float32)
            median = np.tile(median,(self.BRDF_array.shape[0],1))
            cos = np.tile(self.cos_precomputed,(self.BRDF_array.shape[0],1))
            self.X = np.log((self.BRDF_array * cos + self.epsilon) / (median * cos + self.epsilon))
            self.X = self.X.astype(np.float32)
            np.save("./arrays/X.npy",self.X)


            print("Done")


    # only use above hemisphere data
    def generate_valid_mask(self):
        temp = self.BRDF_array[0,:]

        valid_col = []

        if os.path.exists("./arrays/valid_mask_precomputed.npy"):
            print("Reading precomputed valid mask")
            self.valid_mask_precomputed = np.load("./arrays/valid_mask_precomputed.npy")
            self.valid_col_idx = np.load("./arrays/valid_col_idx.npy")
            self.valid_offset = np.load("./arrays/valid_offset.npy")
            self.valid_offset_noNan = np.load("./arrays/valid_offset_noNan.npy")



        else:
            n_invalid = 0
            self.valid_mask_precomputed = np.zeros(self.size,dtype=np.float32)
            self.valid_offset = np.zeros(self.size,dtype=np.float32)
            self.valid_offset = np.where(self.valid_offset == 0, np.NAN, np.NAN).astype(np.float32)
            print("Constructing precomputed valid mask")
            for i in tqdm(range(self.size)):
                if temp[i] < 0.0:
                    self.valid_mask_precomputed[i] = 0.0
                    n_invalid += 1
                else:
                    self.valid_mask_precomputed[i] = 1.0
                    valid_col.append(i)
                    self.valid_offset[i] = n_invalid

            self.valid_col_idx = np.array(valid_col)
            self.valid_mask_precomputed = self.valid_mask_precomputed.astype(np.bool_)
            self.valid_offset_noNan = self.valid_offset[self.valid_col_idx]

            np.save("./arrays/valid_offset.npy",self.valid_offset)
            np.save("./arrays/valid_offset_noNan.npy",self.valid_offset_noNan)
            np.save("./arrays/valid_mask_precomputed.npy",self.valid_mask_precomputed)
            np.save("./arrays/valid_col_idx.npy",self.valid_col_idx)

        self.valid_size = self.valid_col_idx.size


    def generate_cos_weight(self):
        if os.path.exists("./arrays/cos_precomputed.npy"):
            print("Reading precomputed cos")
            self.cos_precomputed = np.load("./arrays/cos_precomputed.npy")

        else:
            self.cos_precomputed = np.zeros(self.size,dtype=np.float32)
            print("Constructing precomputed cosine weight")
            for i in tqdm(range(self.size)):
                result = merl.get_half_diff_coord_from_index(i)
                theta_h, theta_d, phi_d = result[0], result[1], result[2]
                phi_h = 0.0

                wd = util.to_cartesian(theta_d,phi_d)
                wh = util.to_cartesian(theta_h,phi_h)

                wi,wo = util.getwiwo(wh,wd)

                self.cos_precomputed[i] = max(wi[2][0] * wo[2][0], self.epsilon)

            self.cos_precomputed = self.cos_precomputed.astype(np.float32)

            np.save("./arrays/cos_precomputed.npy",self.cos_precomputed)


    def extract_PC(self):


        if os.path.exists("./arrays/mean.npy"):
            self.mean = np.load("./arrays/mean.npy")
            print("Reading mean")
        else:
            self.mean = np.mean(self.X, axis=0)
            np.save("./arrays/mean.npy",self.mean)
            print("Computing mean")


        if os.path.exists("./arrays/scaled_PC.npy"):
            print("Reading principal component")
            self.scaled_pc = np.load("./arrays/scaled_PC.npy")
        else:
            print("Extracting principal component")

            mean = np.tile(self.mean,(self.BRDF_array.shape[0],1)).astype(np.float32)

            temp = self.X - mean
            temp = temp[:,self.valid_col_idx]

            result = np.linalg.svd(temp,False)

            np.save("./arrays/SVD_Vh.npy",result.Vh)

            matrix_s = np.eye(result.S.size)
            for i in range(result.S.size):
                matrix_s[i,i] *= result.S[i]

            np.save("./arrays/SVD_S.npy",matrix_s)
            np.save("./arrays/scaled_PC.npy", result.Vh.T @ matrix_s)




    #
    def convert_from_validIdx_to_fullIdx(self, idx):
        return int(idx + self.valid_offset_noNan[idx])

    def convert_from_fullIdx_to_validIdx(self, idx):
        assert self.valid_offset[idx] != np.NAN
        return int(idx - self.valid_offset[idx])

    def add_point(self,flatten_idx_list : list,idxes_list : list):
        assert len(flatten_idx_list) == len(idxes_list)
        if len(flatten_idx_list) == 0:
            t = np.argmax(np.linalg.norm(self.scaled_pc, axis=1))
            idx_full = self.convert_from_validIdx_to_fullIdx(t)
            flatten_idx_list.append(idx_full)
            idxes_list.append(np.array(merl.get_half_diff_idxes_from_index(idx_full)))

        else:
            niter = 1000
            rng = np.random.default_rng()

            k_min = sys.float_info.max
            n_min = np.NAN

            for i in range(niter):
                n = rng.integers(0, self.valid_col_idx.size)
                flatten_idx_list_test = copy.deepcopy(flatten_idx_list)
                flatten_idx_list_test.append(self.convert_from_validIdx_to_fullIdx(n))

                result = self.get_conditional_number(np.array(flatten_idx_list_test))

                if result < k_min:
                    k_min = result
                    n_min = n

            flatten_idx_list.append(self.convert_from_validIdx_to_fullIdx(n_min))
            idxes_list.append(np.array(merl.get_half_diff_idxes_from_index(flatten_idx_list[-1])))

        return flatten_idx_list, idxes_list

    def find_optimal_directions(self,n_dir):

        np.seterr('raise')

        rng = np.random.default_rng()

        grad_dirs = np.array([[1, 0, 0],  # Gradient directions
                             [-1, 0, 0],
                             [0, 1, 0],
                             [0, -1, 0],
                             [0, 0, 1],
                             [0, 0, -1]])

        upper_bound = np.array([89,89,179]) # used to clip sample
        lower_bound = np.array([0,0,0]) # used to clip sample


        # flatten_idx_list contains full idx!!!
        flatten_idx_list = []
        idxes_list = []


        max_iter = 20000


        self.scaled_pc = np.load("./arrays/scaled_PC.npy")
        self.mean = np.load("./arrays/mean.npy")


        while len(flatten_idx_list) != n_dir:
            flatten_idx_list, idxes_list = self.add_point(flatten_idx_list, idxes_list)
            n_iter = 0
            converged = False
            converge_percentage = 0.001
            last_conditional_num = np.inf
            n_meet_convergence = 0
            step_size = 3

            while not converged and n_iter < max_iter:
                flatten_idx_list_current_run = copy.deepcopy(flatten_idx_list)
                idxes_list_current_run = copy.deepcopy(idxes_list)

                best_flatten_idx_list_current_run = copy.deepcopy(flatten_idx_list_current_run)
                best_idxes_list_current_run = copy.deepcopy(idxes_list_current_run)


                order = np.random.permutation(len(flatten_idx_list_current_run))
                best_condition_num = self.get_conditional_number(np.array(flatten_idx_list_current_run))
                for i in order:


                    for d in grad_dirs:
                        flatten_idx_list_new, idxes_list_new = copy.deepcopy(flatten_idx_list_current_run), copy.deepcopy(idxes_list_current_run)
                        n_new, point_new = flatten_idx_list_new[i], idxes_list_new[i]

                        # check if these changes really change the list
                        point_new += d * step_size
                        point_new = self.clip_idx(point_new)
                        point_new = point_new.astype(np.int32)

                        n_new = merl.get_index_from_half_diff_idxes(point_new[0],point_new[1],point_new[2])
                        flatten_idx_list_new[i] = n_new

                        # If this permutation is valid
                        if self.valid_mask_precomputed[n_new] == 1.0:
                            current_condition = self.get_conditional_number(np.array(flatten_idx_list_new))
                            if best_condition_num > current_condition:
                                best_condition_num = current_condition
                                best_flatten_idx_list_current_run = copy.deepcopy(flatten_idx_list_new)
                                best_idxes_list_current_run = copy.deepcopy(idxes_list_new)


                if last_conditional_num!=np.inf and (last_conditional_num - best_condition_num) / last_conditional_num * 100 <= converge_percentage:
                    step_size = 1
                    n_meet_convergence += 1
                    if n_meet_convergence > 2:
                        converged = True
                        flatten_idx_list = copy.deepcopy(best_flatten_idx_list_current_run)
                        idxes_list = copy.deepcopy(best_idxes_list_current_run)

                else:
                    n_meet_convergence = 0

                n_iter += 1

                last_conditional_num = best_condition_num

        self.save_dir_related_info(flatten_idx_list, idxes_list)

        return flatten_idx_list, idxes_list


    def save_dir_related_info(self,flatten_idx_list, idxes_list):
        np.save("./direction/flatten_idx_list.npy", np.array(flatten_idx_list))
        np.save("./direction/idxes_list.npy", np.array(idxes_list))

        n_list = np.array(flatten_idx_list).astype(np.int32)
        n_list_valid = n_list - self.valid_offset[n_list]
        n_list_valid = n_list_valid.astype(np.int32)
        q = self.scaled_pc[n_list_valid, :]

        np.save("./direction/reduced_scaled_PC.npy", q)

        mean = self.mean[n_list.astype(np.int32)]
        np.save("./direction/reduced_mean.npy",mean)




    def clip_idx(self, idxes : np.ndarray):
        # index_list must follow t_h,t_d,p_d
        upper_bound = [89,89,179]
        lower_bound = [0,0,0]

        for i in range(3):
            idxes[i] = np.clip(idxes[i], lower_bound[i], upper_bound[i])



        return idxes



    def get_gradient(self, params2 : [], params1 : [], which, n_list , idx_to_replace):
        t_h_2, t_d_2, p_d_2 = params2[0], params2[1] , params2[2]
        t_h_1, t_d_1, p_d_1 = params1[0], params1[1] , params1[2]

        idx2 = merl.get_index_from_half_diff_idxes(params2[0], params2[1] , params2[2])
        idx1 = merl.get_index_from_half_diff_idxes(params1[0], params1[1] , params1[2])

        n_list2 = np.copy(n_list)
        n_list2[idx_to_replace] = idx2

        n_list1 = np.copy(n_list)
        n_list1[idx_to_replace] = idx1

        Q2 = self.scaled_pc[n_list2,:]
        Q1 = self.scaled_pc[n_list1,:]

        svd_result2 = np.linalg.svd(Q2,full_matrices=False)
        svd_result1 = np.linalg.svd(Q1,full_matrices=False)

        k2 = svd_result2.S[0] / svd_result2.S[-1]
        k1 = svd_result1.S[0] / svd_result1.S[-1]

        return (k2 - k1) / (params2[which] - params1[which])


    # n_list contains full idx
    def get_conditional_number(self, n_list : np.ndarray):
        # need to make sure all idx in n_list is valid
        n_list = n_list - self.valid_offset[n_list]
        n_list = n_list.astype(np.int32)
        Q = self.scaled_pc[n_list, :]
        svd_result = np.linalg.svd(Q, full_matrices=False)
        k = svd_result.S[0] / svd_result.S[-1]
        return k



    def test_idx(self):
        for i in range(1000):
            valid_brdf = self.BRDF_array[:,self.valid_col_idx]


            rng = np.random.default_rng()
            u, v = rng.random(2)
            wi = util.random_uniform_hemisphere(u, v)
            theta_i, phi_i = util.to_spherical(wi)
            u, v = rng.random(2)
            wo = util.random_uniform_hemisphere(u, v)
            theta_o, phi_o = util.to_spherical(wo)

            result = merl.convert_to_hd(theta_i, phi_i, theta_o, phi_o)
            theta_h, phi_h, theta_d, phi_d = result[0], result[1], result[2], result[3]

            result = merl.get_index_from_hall_diff_coords(theta_h,theta_d,phi_d)

            result_valid = self.convert_from_fullIdx_to_validIdx(result)

            slice_valid = valid_brdf[:,result_valid]
            slice_full = self.BRDF_array[:,result]

            print("?")


class linear_combination_brdf:
    reference_merl: MERL_Collection
    flatten_idx_list: np.ndarray
    idxes_list: np.ndarray
    reduced_scaled_PC: np.ndarray
    reduced_mean: np.ndarray

    # should be 3 * n_dir
    observed_rgb : np.ndarray

    c : np.ndarray

    BRDF_array : np.ndarray

    eta : float

    def __init__(self, has_data : bool, find_direction:bool = False):
        if not has_data:
            self.reference_merl = MERL_Collection()
            self.reference_merl.generate_valid_mask()
            self.reference_merl.generate_cos_weight()
            self.reference_merl.get_reference()
            self.reference_merl.extract_PC()
            if find_direction:
                self.flatten_idx_list, self.idxes_list = self.reference_merl.find_optimal_directions(20)
            self.BRDF_array = self.reference_merl.BRDF_array
        else:
            self.BRDF_array = np.load("./arrays/matrix.npy")
        self.read_direction_info()
        self.eta = 40
        self.c = np.zeros((3,300))




    def read_direction_info(self):
        self.flatten_idx_list = np.load("./direction/flatten_idx_list.npy")
        self.idxes_list = np.load("./direction/idxes_list.npy")

        self.reduced_scaled_PC = np.load("./direction/reduced_scaled_PC.npy")
        self.reduced_mean = np.load("./direction/reduced_mean.npy")



    # per-channel reconstruction
    def reconstruction(self):
        # reduced Q has the shape of ndir * 300
        Q = self.reduced_scaled_PC
        for i in range(3):
            x = self.observed_rgb[i,:]
            # Q.T @ Q + eta * I -> 300 * 300
            # Q.T -> 300 * ndir
            # (x-mean) -> ndir * 1

            self.c[i,:] = np.linalg.inv((Q.T @ Q + 40 * np.eye(Q.shape[1]))) @ Q.T @ (x - self.reduced_mean)

    def eval_io_rgb(self,wi,wo):
        theta_i,phi_i = util.to_spherical(wi)
        theta_o,phi_o = util.to_spherical(wo)

        result = merl.convert_to_hd(theta_i,phi_i,theta_o,phi_o)

        theta_h,theta_d,phi_d = result[0],result[1],result[2]

        value = self.eval_hd_rgb(theta_h,theta_d,phi_d)
        return value

    def eval_hd_rgb(self, theta_h, theta_d, phi_d):
        full_idx = merl.get_index_from_hall_diff_coords(theta_h, theta_d, phi_d)
        col = self.BRDF_array[:, full_idx]

        value = np.zeros(3)

        for i in range(3):
            value[i] = np.dot(col,self.c[i,:])

        return value

    def eval_io_channel(self,wi,wo,channel_idx:int):
        theta_i, phi_i = util.to_spherical(wi)
        theta_o, phi_o = util.to_spherical(wo)

        result = merl.convert_to_hd(theta_i, phi_i, theta_o, phi_o)

        theta_h, theta_d, phi_d = result[0], result[1], result[2]

        value = self.eval_hd_channel(theta_h,theta_d,phi_d,channel_idx)
        return value

    def eval_hd_channel(self,theta_h, theta_d, phi_d, channel_idx : int):
        full_idx = merl.get_index_from_hall_diff_coords(theta_h, theta_d, phi_d)
        col = self.BRDF_array[:, full_idx]
        value = np.dot(col, self.c[channel_idx,:])
        return value










if __name__ == "__main__":
    os.chdir("/home/yuan/school/cp/final_project/code")

    #test = MERL_Collection()

    #test.generate_valid_mask()
    #test.generate_cos_weight()
    #test.get_reference()
    #test.extract_PC()

    #test.find_optimal_directions(10)

    test = linear_combination_brdf(True)

    print("Done")

                

    