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
    median : np.ndarray #median is the median after cos weight?

    ## valid mask
    valid_mask_precomputed : np.ndarray
    valid_col_idx : np.ndarray
    valid_offset : np.ndarray # offset used to map valid array to full array (110xxx -> 145xxx). It has a length of 145xxx, with invalid locations set to np.nan
    valid_offset_noNan : np.ndarray

    scaled_pc : np.ndarray
    scaled_pc_pinv : np.ndarray #psudo inverse of scaled pc

    epsilon = 0.001

    half_diff_meshgrid : None

    # p_d : list
    # theta_d : list
    # theta_h : list


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


        # #create meshgrid for half diff space
        # phi_d_ = np.linspace(0,179,180, endpoint=True).astype(np.uint8)
        # theta_d_ = np.linspace(0,89,90, endpoint=True).astype(np.uint8)
        # theta_h_ = np.linspace(0,89,90, endpoint=True).astype(np.uint8)
        #
        # self.p_d, self.theta_d, self.theta_h = np.meshgrid(phi_d_,theta_d_,theta_h_, indexing='ij')


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
            print("Reading median")
            self.median = np.load("./arrays/median.npy")
        else:
            # it doesn't matter for cos and median as it only depends on one direction

            # In the original code, median is computed after BRDF * cos
            cos = np.tile(self.cos_precomputed,(self.BRDF_array.shape[0],1))

            print("Constructing observation matrix X")
            median = np.median(self.BRDF_array * cos,axis=0).astype(np.float32)

            np.save("./arrays/median.npy",median)

            median = np.tile(median,(self.BRDF_array.shape[0],1))

            self.X = np.log((self.BRDF_array * cos + self.epsilon) / (median + self.epsilon))
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

            temp2 = np.load("/home/yuan/school/cp/code/data/X_minus_mean.npy")

            result2 = np.linalg.svd(temp2,full_matrices=False)

            np.save("./arrays/SVD_Vh.npy",result.Vh)

            matrix_s = np.eye(result.S.size)
            for i in range(result.S.size):
                matrix_s[i,i] *= result.S[i]

            np.save("./arrays/SVD_S.npy",matrix_s)
            np.save("./arrays/scaled_PC.npy", result.Vh.T @ matrix_s)


    def regularized_inverse(self,A, eta):

        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        Ut = U.T
        V = Vt.T
        Sinv = np.diag(s/(s*s+eta))
        A_plus = V @ Sinv @ Ut

        return A_plus

    def get_error_metric(self,n_list: np.ndarray):
        # n_list is the center's direction
        all_direction_idxes = []
        for i in range(n_list.size):
            theta_h_idx, theta_d_idx, phi_d_idx = merl.get_half_diff_idxes_from_index(n_list[i])
            result = self.get_related_n_from_center_direction(theta_h_idx,theta_d_idx,phi_d_idx)
            all_direction_idxes += result

        # ignore noise
        beta = 0


        Y_matrix = self.X - self.mean
        Y_matrix = Y_matrix[:,self.valid_col_idx]


        valid_idxes = self.convert_fulllist_to_validlist(np.array(all_direction_idxes))


        S = self.convert_nlist_to_selection_matrix(np.array(all_direction_idxes).flatten())

        Q = self.scaled_pc
        #Q_plus = np.linalg.pinv(Q)
        Q_plus = self.scaled_pc_pinv

        #SQ_reg_inv = self.regularized_inverse(S@Q,40)
        SQ_reg_inv = self.regularized_inverse(Q[valid_idxes], 40)


        # This is putting each column of SQ_reg_inv to [idx] column in full matrix. Other colums are all zero
        # Computing SQ_plus S in eq 12
        #temp = SQ_reg_inv @ S
        temp = np.zeros_like(Q_plus)
        temp[:,valid_idxes] = SQ_reg_inv

        # for i in range(SQ_reg_inv.shape[1]):
        #     test2_test[:,valid_idxes[i]] = SQ_reg_inv[:,i]


        co = Q_plus - temp


        sum_ = 0




        print("Computing Error for this setting")
        # Vectorized version of the below for loop
        result = Q @ (co @ Y_matrix.T)
        norms = np.linalg.norm(result,axis=0)
        return np.mean(norms)

        # This is too slow
        # for i in tqdm(range(Y_matrix.shape[0])):
        #
        #     tmp = co @ Y_matrix[i]
        #
        #     tmp = Q @ tmp
        #
        #     sum_ += np.linalg.norm(tmp)
        #
        # sum_ /= Y_matrix.shape[0]
        #
        # return sum_


    def convert_fulllist_to_validlist(self, full_list : np.ndarray):
        full_list = full_list.flatten().astype(np.int32)
        valid_list = full_list - self.valid_offset[full_list]
        return valid_list.astype(np.int32)


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
            niter = 3000
            rng = np.random.default_rng()

            k_min = sys.float_info.max
            n_min = np.NAN

            for i in range(niter):
                n = rng.integers(0, self.valid_col_idx.size)
                flatten_idx_list_test = copy.deepcopy(flatten_idx_list)
                flatten_idx_list_test.append(self.convert_from_validIdx_to_fullIdx(n))

                result = self.get_conditional_number_primary(np.array(flatten_idx_list_test))

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

            if len(flatten_idx_list) == 1:
                continue


            n_iter = 0
            converged = False
            converge_percentage = 0.001
            last_conditional_num = np.inf
            n_meet_convergence = 0
            step_size = 3

            flatten_idx_list_current_run = copy.deepcopy(flatten_idx_list)
            idxes_list_current_run = copy.deepcopy(idxes_list)


            while not converged and n_iter < max_iter:


                order = np.random.permutation(len(flatten_idx_list_current_run))
                best_condition_num = self.get_conditional_number_primary(np.array(flatten_idx_list_current_run))
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
                        if self.valid_mask_precomputed[n_new] == 1.0 and np.unique(np.array(flatten_idx_list_new)).size == len(flatten_idx_list_new):
                            current_condition = self.get_conditional_number_primary(np.array(flatten_idx_list_new))
                            if best_condition_num > current_condition:
                                best_condition_num = current_condition
                                flatten_idx_list_current_run = copy.deepcopy(flatten_idx_list_new)
                                idxes_list_current_run = copy.deepcopy(idxes_list_new)

                if last_conditional_num != np.inf:
                    print("Iteration ", n_iter)
                    print("Step size", step_size)
                    print("Current cond num", best_condition_num)
                    print(f"cond num reduction {last_conditional_num - best_condition_num}, {(last_conditional_num - best_condition_num) / last_conditional_num * 100}")


                if last_conditional_num!=np.inf and (last_conditional_num - best_condition_num) / last_conditional_num * 100 <= converge_percentage:
                    step_size = 1
                    n_meet_convergence += 1
                    if n_meet_convergence > 2:
                        converged = True
                        flatten_idx_list = copy.deepcopy(flatten_idx_list_current_run)
                        idxes_list = copy.deepcopy(idxes_list_current_run)

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


    def save_dir_test(self):
        coords = np.array([
            [3,12,28],
            [63,19,89],
            [5,77,77],
            [2,60,180],
            [15,4,130],
            [1,6,37],
            [2,79,110],
            [39,76,89],
            [0,71,104],
            [5,75,180]
        ])

        f = np.zeros(10,dtype= np.int32)
        idd = np.zeros((10,3),dtype=np.int32)




        for i in range(10):
            theta_h = np.pi * coords[i][0] / 180
            theta_d = np.pi * coords[i][1] / 180
            phi_d = np.pi * coords[i][2] / 180

            idx = merl.get_index_from_hall_diff_coords(theta_h,theta_d,phi_d)

            f[i] = idx

            idxes = merl.get_half_diff_idxes_from_index(idx)
            idd[i][0] = idxes[0]
            idd[i][1] = idxes[1]
            idd[i][2] = idxes[2]

        np.save("./direction/flatten_idx_list_test.npy", f)
        np.save("./direction/idxes_list_test.npy", idd)

        n_list_valid = f - self.valid_offset[f]
        n_list_valid = n_list_valid.astype(np.int32)
        q = self.scaled_pc[n_list_valid, :]

        np.save("./direction/reduced_scaled_PC_test.npy", q)

        mean = self.mean[f.astype(np.int32)]
        np.save("./direction/reduced_mean_test.npy", mean)


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


    def get_conditional_number_primary(self,n_list : np.ndarray):
        n_list = n_list - self.valid_offset[n_list]
        n_list = n_list.astype(np.int32)
        Q = self.scaled_pc[n_list, :]

        (p,n) = Q.shape
        if p < n:
            Q = Q[:,0:p]

        return np.linalg.cond(Q)


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
    flatten_idx_list: np.ndarray # n_dir
    idxes_list: np.ndarray    # n_dir * 3
    reduced_scaled_PC: np.ndarray # n_dir * 300
    reduced_mean: np.ndarray # n_dir

    # should be 3 * n_dir
    observed_rgb : np.ndarray

    c: np.ndarray

    BRDF_array: np.ndarray

    eta: float

    n_dir: int

    # only valid entries
    # shoule be 3 * 115xxx
    valid_rgb : np.ndarray

    def __init__(self, has_data : bool, find_direction:bool = False):
        if not has_data:
            self.reference_merl = MERL_Collection()
            self.reference_merl.generate_valid_mask()
            self.reference_merl.generate_cos_weight()
            self.reference_merl.get_reference()
            self.reference_merl.extract_PC()
            #self.reference_merl.save_dir_test()
            if find_direction:
                self.flatten_idx_list, self.idxes_list = self.reference_merl.find_optimal_directions(10)
            self.BRDF_array = self.reference_merl.BRDF_array
        else:
            self.BRDF_array = np.load("./arrays/matrix.npy")
        self.read_direction_info()
        self.eta = 40


    def initialize_merl_test_data(self):
        mat = merl.MERL_BRDF(directory_path + "green-plastic.binary")

        for i in range(self.n_dir):
            theta_h,theta_d,phi_d = int(self.idxes_list[i][0]),int(self.idxes_list[i][1]),int(self.idxes_list[i][2])
            value = mat.look_up_hdidx(theta_h,theta_d,phi_d)
            self.observed_rgb[:,i] = np.array(value)

        self.reconstruction()

        self.compute_RMS(mat)

        print("Done")





    def read_direction_info(self):
        self.flatten_idx_list = np.load("./direction/flatten_idx_list.npy")
        self.idxes_list = np.load("./direction/idxes_list.npy")

        self.reduced_scaled_PC = np.load("./direction/reduced_scaled_PC.npy")
        self.reduced_mean = np.load("./direction/reduced_mean.npy")

        self.n_dir = self.idxes_list.shape[0]
        self.observed_rgb = np.zeros((3,self.n_dir))
        self.valid_rgb = np.zeros((3,self.reference_merl.valid_col_idx.size))



    # per-channel reconstruction
    def reconstruction(self):
        # Try
        # Only use for n_dir principal components?


        # reduced Q has the shape of ndir * 300
        Q_tilde = self.reduced_scaled_PC

        # self.c = np.zeros((3,self.n_dir))
        # Q_tilde = Q_tilde[:,0:self.n_dir]

        self.c = np.zeros((3,300))
        Q_tilde = Q_tilde

        for i in range(3):
            data = self.observed_rgb[i,:]

            # map the original data
            data = np.log((data * self.reference_merl.cos_precomputed[self.flatten_idx_list] + self.reference_merl.epsilon) / (self.reference_merl.median[self.flatten_idx_list] + self.reference_merl.epsilon) )

            # Q_tilde.T @ Q_tilde + eta * I -> n_dir * n_dir
            # Q_tilde.T -> n_dir * ndir
            # (x-mean) -> n_dir * 1

            # The same as below
            # b = data - self.reduced_mean
            # U,s,Vt = np.linalg.svd(Q_tilde,full_matrices=False)
            # Ut = U.T
            # V = Vt.T
            # Sinv = np.diag(s/(s*s+self.eta))
            # x = V @ Sinv @ Ut @ b
            # self.c[i,:] = x

            self.c[i,:] = np.linalg.inv((Q_tilde.T @ Q_tilde + 40 * np.eye(Q_tilde.shape[1]))) @ Q_tilde.T @ (data - self.reduced_mean)

        # Q -> 115xxx * 300
        # c -> 300 * 1
        # mean -> 115xxx * 1


        mean = self.reference_merl.mean[self.reference_merl.valid_col_idx]
        #Q = self.reference_merl.scaled_pc[:,0:self.n_dir]
        Q = self.reference_merl.scaled_pc

        for i in range(3):
            # We use cosine map, need to divide by it
            self.valid_rgb[i, :] = (Q @ self.c[i, :]) + mean

        self.unmap_brdf()

        print("Done")


    def unmap_brdf(self):
        for i in range(3):
            channel = self.valid_rgb[i,:]

            unmapped_result = np.exp(channel) * (self.reference_merl.median[self.reference_merl.valid_col_idx] + self.reference_merl.epsilon) - self.reference_merl.epsilon

            unmapped_result /= self.reference_merl.cos_precomputed[self.reference_merl.valid_col_idx]

            self.valid_rgb[i,:] = unmapped_result

        print("Done")


    def eval_io_rgb(self,wi,wo):
        theta_i,phi_i = util.to_spherical(wi)
        theta_o,phi_o = util.to_spherical(wo)

        result = merl.convert_to_hd(theta_i,phi_i,theta_o,phi_o)

        theta_h,theta_d,phi_d = result[0],result[1],result[2]

        value = self.eval_hd_rgb(theta_h,theta_d,phi_d)
        return value

    def eval_hd_rgb(self, theta_h, theta_d, phi_d):
        full_idx = merl.get_index_from_hall_diff_coords(theta_h, theta_d, phi_d)

        valid_idx = self.reference_merl.convert_from_fullIdx_to_validIdx(full_idx)

        value = self.valid_rgb[:, valid_idx]

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
        valid_idx = self.reference_merl.convert_from_fullIdx_to_validIdx(full_idx)
        value = self.valid_rgb[channel_idx,valid_idx]
        return value

    def eval_hdidx(self,theta_h_idx, theta_d_idx, phi_d_idx):
        full_idx = merl.get_index_from_half_diff_idxes(theta_h_idx,theta_d_idx,phi_d_idx)
        valid_idx = self.reference_merl.convert_from_fullIdx_to_validIdx(full_idx)
        value = self.valid_rgb[:, valid_idx]
        return value


    def write_to_merl_format(self):
        r_scale = 1500
        g_scale = 1500 / 1.15
        b_scale = 1500 / 1.66


        with open("./material.bsdf","wb") as f:
            dim = np.array([180,90,90],dtype=np.int32)
            byte = dim.tobytes()
            f.write(byte)

            # generate full rgb
            full_rgb = np.zeros((3,180*90*90),dtype=np.float64)

            full_rgb[:,self.reference_merl.valid_col_idx] = self.valid_rgb

            full_rgb = np.clip(full_rgb,0.0,None)

            full_rgb[0,:] *= r_scale
            full_rgb[1,:] *= g_scale
            full_rgb[2,:] *= b_scale

            byte = full_rgb.tobytes()
            f.write(byte)


    def compute_RMS(self,mat:merl.MERL_BRDF):
        gt_valid_rgb = np.zeros_like(self.valid_rgb)

        r_val = np.array(mat.r_channel_unscaled) * mat.r_scale
        g_val = np.array(mat.g_channel_unscaled) * mat.g_scale
        b_val = np.array(mat.b_channel_unscaled) * mat.b_scale

        gt_valid_rgb[0,:] = r_val[self.reference_merl.valid_col_idx]
        gt_valid_rgb[1,:] = g_val[self.reference_merl.valid_col_idx]
        gt_valid_rgb[2,:] = b_val[self.reference_merl.valid_col_idx]

        rms = np.sqrt(np.mean((gt_valid_rgb - self.valid_rgb) ** 2))

        print(rms)


class gt_data:
    median : np.ndarray
    relative_offset : np.ndarray # mean
    scaled_pc : np.ndarray
    cos_precomputed: np.ndarray

    def __init__(self):
        self.median = np.load("/home/yuan/school/cp/code/data/Median.npy")
        self.relative_offset = np.load("/home/yuan/school/cp/code/data/RelativeOffset.npy")
        self.scaled_pc= np.load("/home/yuan/school/cp/code/data/ScaledEigenvectors.npy")
        self.cos_precomputed = np.load("/home/yuan/school/cp/code/data/CosineMap.npy")

        print("Reading gt data")





if __name__ == "__main__":
    os.chdir("/home/yuan/school/cp/final_project/code")

    #data = gt_data()

    #test = MERL_Collection()

    #test.generate_valid_mask()
    #test.generate_cos_weight()
    #test.get_reference()
    #test.extract_PC()

    #test.find_optimal_directions(10)

    test = linear_combination_brdf(False,False)
    test.initialize_merl_test_data()

    print("Done")

                

    