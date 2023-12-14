import numpy as np   
import scipy

import os
import sys

import merl

import util

from tqdm import tqdm

directory_path = "/home/yuan/school/cp/final_project/BRDFDatabase/brdfs/"

class MERL_Collection:
    
    materials = []
    

    #should be 300 * 1458000 in dimension
    BRDF_array : np.ndarray
    X : np.ndarray
    cos_precomputed : np.ndarray

    scaled_pc : np.ndarray

    epsilon = 0.001

    half_diff_meshgrid : None

    p_d : list
    theta_d : list
    theta_h : list


    size : int

    
    def __init__(self) -> None:

        if os.path.exists('./arrays/matrix.npy'):
            print("Reading BRDF matrix")
            self.BRDF_array = np.load('./arrays/matrix.npy')


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
        result = merl.get_half_diff_coord_from_index(idx)

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
            self.X = np.log(   (self.BRDF_array * cos + self.epsilon) / (median * cos + self.epsilon))
            self.X = self.X.astype(np.float32)
            np.save("./arrays/X.npy",self.X)


            print("Done")


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
        mean = np.mean(self.X,axis=0)

        mean = np.tile(mean,(self.BRDF_array.shape[0],1)).astype(np.float32)



        result = np.linalg.svd(self.X - mean,False)


        print("Done")


    def find_optimal_directions(self,n_dir):

        n_list = []


        self.scaled_pc = np.load("./arrays/scaled_PC.npy")

        k_min = sys.float_info.max
        n_min = None

        step_length = 3

        r = 1

        # step 1
        rng = np.random.default_rng()
        for i in range(20):
            n = rng.integers(0,self.size,size = 1)
            Q_truncated = self.scaled_pc[n,:]
            result = np.linalg.svd(Q_truncated,full_matrices=False)
            k = result.U[0,0] / result.U[-1,-1]
            if k < k_min:
                k_min = k
                n_min = n

        n_list.append(n_min)


        upper_bound = [89,89,179]
        lower_bound = [0,0,0]


        # repeat until find n_dir points
        while(len(n_list) < n_dir):


            n_new = n_list[rng.integers(0,len(n_list))]
            k_last_it = sys.float_info.max
            tolerance = 1e-5

            # step 2 : compute gradient
            while True:
                current_idx = merl.get_half_diff_idxes_from_index(n_new)

                current_idx = np.array(current_idx).astype(np.int32)

                t_h_idx,t_d_idx,p_d_idx = current_idx[0],current_idx[1],current_idx[2]

                grad = np.zeros(3)

                for i in range(3):
                    params2 = np.copy(current_idx)
                    if params2[i] + 1 < upper_bound[i]:
                        params2[i] = params2[i] + 1

                    params1 = np.copy(current_idx)
                    if params2[i] - 1 > lower_bound[i]:
                        params1[i] = (params1[i] - 1)

                    g = self.get_gradient(params2,params1,i)

                    grad[i] = g

                #move along gradient direction (plus or substract?)
                current_idx += grad * step_length * current_idx
                current_idx = self.clip_idx(current_idx)

                n_new = merl.get_index_from_half_diff_idxes(current_idx[0],current_idx[1],current_idx[2])

                # termination condition?
                Q_truncated = self.scaled_pc[n_new, :]
                result = np.linalg.svd(Q_truncated, full_matrices=False)
                k_new = result.U[0, 0] / result.U[-1, -1]

                if np.linalg.norm(k_new - k_last_it) > tolerance:
                    k_last_it = k_new
                    continue
                elif step_length == 3:
                    step_length = 1
                    tolerance = 1e-7
                    continue
                else:
                    break
            # step 4 add points
            r += 1



    def clip_idx(self, idxes : np.ndarray):
        # index_list must follow t_h,t_d,p_d
        upper_bound = [89,89,179]
        lower_bound = [0,0,0]

        for i in range(3):
            idxes[i] = np.clip(idxes[i], lower_bound[i], upper_bound[i])

        return idxes

    def get_gradient(self, params2 : [], params1 : [], which):
        t_h_2, t_d_2, p_d_2 = params2[0], params2[1] , params2[2]
        t_h_1, t_d_1, p_d_1 = params1[0], params1[1] , params1[2]

        idx2 = merl.get_index_from_half_diff_idxes(params2[0], params2[1] , params2[2])
        idx1 = merl.get_index_from_half_diff_idxes(params1[0], params1[1] , params1[2])

        Q2 = self.scaled_pc[idx2,:]
        Q1 = self.scaled_pc[idx1,:]

        svd_result2 = np.linalg.svd(Q2,full_matrices=False)
        svd_result1 = np.linalg.svd(Q1,full_matrices=False)

        k2 = svd_result2.U[0, 0] / svd_result2.U[-1, -1]
        k1 = svd_result1.U[0, 0] / svd_result1.U[-1, -1]

        return (k2 - k1) / (params2[which] - params1[which])





if __name__ == "__main__":
    os.chdir("/home/yuan/school/cp/final_project/code")

    test = MERL_Collection()

    # test.generate_cos_weight()
    # test.get_reference()
    # test.extract_PC()

    test.find_optimal_directions(10)

    print("Done")

                

    