# From optimized dir code


n_list = []





# repeat until find n_dir points
whil e(len(n_list) < n_dir):
    idx_to_replace = rng.integers(0 ,len(n_list))

    n_new = n_list[idx_to_replace]
    k_last_it = sys.float_info.max
    tolerance = 3
    step_length = 3
    current_idx = merl.get_half_diff_idxes_from_index(n_new)

    idx_history = []

    # step 2 : compute gradient
    while True:
        current_idx = np.array(current_idx).astype(np.float32)

        grad = np.zeros(3)

        for i in range(3):
            params2 = np.copy(current_idx)
            if params2[i] + 1 < upper_bound[i]:
                params2[i] = params2[i] + 1
                temp = merl.get_index_from_half_diff_idxes(params2[0] ,params2[1] ,params2[2])
                if self.valid_mask_precomputed[temp] != 1.0:
                    params2 = np.copy(current_idx)

            params1 = np.copy(current_idx)
            if params1[i] - 1 > lower_bound[i]:
                params1[i] = (params1[i] - 1)
                temp = merl.get_index_from_half_diff_idxes(params1[0] ,params1[1] ,params1[2])
                if self.valid_mask_precomputed[temp] != 1.0:
                    params1 = np.copy(current_idx)

            g = self.get_gradient(params2.astype(np.int32) ,params1.astype(np.int32) ,i, np.array(n_list), idx_to_replace)

            grad[i] = g

        # Move integer steps?
        int_grad = grad / np.abs(grad)

        # move along gradient direction (plus or substract?)
        current_idx -= int_grad * step_length
        current_idx = self.clip_idx(current_idx)