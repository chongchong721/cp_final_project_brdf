import numpy as np   
import scipy

import os
import sys

import merl

direactory_path = "/home/yuan/school/cp/final_project/BRDFDatabase/brdfs/"

class MERL_Collection:
    
    materials = []
    

    
    def __init__(self) -> None:
        size = 0


        with open("./name_list") as f:
            lines = f.readlines()
            assert(len(lines) == 100)

            for line in lines:
                # there is an extra \n
                line = direactory_path + line[:-1]
                mat = merl.MERL_BRDF(line)
                self.materials.append(mat)
                size = mat.m_size





if __name__ == "__main__":
    os.chdir("/home/yuan/school/cp/final_project/code")

    test = MERL_Collection()

    print("Done")

                

    