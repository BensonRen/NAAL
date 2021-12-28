def smooth(scalars, weight=SMOOTH_WEIGHT):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

##########################################
# Plotting various MSE metrics together #
##########################################
def plot_various_MSE(x_label, MSE_test_mse, RD_test_mse, MSE_train_mse, RD_train_mse, VAR_test_mse, VAR_train_mse, plot_var, save_name):
    f = plt.figure(figsize=[8, 4])
    #ax1 = plt.subplot(211)
    plt.plot(x_label, MSE_test_mse, '-x', c='tab:blue', label='MSE test')
    plt.plot(x_label, RD_test_mse, '-x', c='tab:orange', label='RD test')
    plt.plot(x_label, MSE_train_mse, '--x', c='tab:blue', linewidth=2, label='MSE train')
    plt.plot(x_label, RD_train_mse, '--x', c='tab:orange', linewidth=2, label='RD train')
    if plot_var:
        plt.plot(x_label, VAR_test_mse, '-x', c='tab:green', label='VAR test')
        plt.plot(x_label, VAR_train_mse, '--x', c='tab:green', linewidth=2, label='VAR train')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('num_train')
    plt.ylabel('MSE')
    plt.title(os.path.basename(save_name))
    plt.savefig(save_name+'.png')
    plt.clf()

###################################
# Plotting AL/Raondom Ratio plot #
###################################
def plot_ratio(num, denom, x_value, ylabel, save_name, x_label):
    # Get the MSE/Random ratio
    test_mse_ratio = num / denom
    test_mse_ratio = np.log(test_mse_ratio)
    percent_better = np.mean(test_mse_ratio < 0)
    #test_mse_ratio[test_mse_ratio < 1] = -np.log(1/test_mse_ratio[test_mse_ratio < 1])
    #test_mse_ratio[test_mse_ratio >= 1] = np.log(test_mse_ratio[test_mse_ratio >= 1])
    # Plot the ratio figure
    f = plt.figure(figsize=[8, 4])
    for i in range(num_trails):
        plt.plot(x_value, test_mse_ratio[:, i],'x')
    plt.plot(x_value, np.zeros_like(x_label),'--r', label='ref: {:.1f}%'.format(percent_better*100))
    plt.legend()
    #plt.ylim([-2, 2])
    plt.yscale('log')
    plt.xlabel('num_train')
    plt.ylabel(ylabel)
    plt.title(os.path.basename(save_name))
    plt.savefig(save_name + 'ratio_plot.png')
    plt.clf()

#######################################
# Plotting Test MSE comparison plot #
#######################################
def plot_test_MSE_plot(x_label, MSE_test_mse_mat, Random_test_mse_mat, save_name, VAR_test_mse_mat=None, 
                    NA_test_mse_mat=None, plot_var=True, alpha=0.2, random_folder_provide=None,NAMD_test_mse_mat=None):
    
    if random_folder_provide:
        x_label_rand = get_x_label_from_folder_name(random_folder_provide)
    else:
        x_label_rand = x_label
    # Get the comparison plot for all
    f = plt.figure(figsize=[8, 4])
    # For easy legend
    plt.plot(x_label, MSE_test_mse_mat[:, 0], '-', alpha=alpha,  c='tab:blue', label='AL')
    plt.plot(x_label_rand, Random_test_mse_mat[:, 0], '-', alpha=alpha,  c='tab:orange', label='Random')
    if plot_var:
        plt.plot(x_label, VAR_test_mse_mat[:, 0], '-', alpha=alpha,  c='tab:green', label='VAR')
    if plot_na:
        plt.plot(x_label, NA_test_mse_mat[:, 0], '-', alpha=alpha,  c='tab:brown', label='NA')
    if plot_namd:
        plt.plot(x_label, NAMD_test_mse_mat[:, 0], '-', alpha=alpha,  c='tab:red', label='NAMD')
    for i in range(1, num_trails):
        plt.plot(x_label, MSE_test_mse_mat[:, i], '-', alpha=alpha,  c='tab:blue')
        plt.plot(x_label_rand, Random_test_mse_mat[:, i], '-', alpha=alpha,  c='tab:orange')
        if plot_var:
            plt.plot(x_label, VAR_test_mse_mat[:, i], '-', alpha=alpha,  c='tab:green')
        if plot_na:
            plt.plot(x_label, NA_test_mse_mat[:, i], '-', alpha=alpha,  c='tab:brown')
        if plot_namd:
            plt.plot(x_label, NAMD_test_mse_mat[:, i], '-', alpha=alpha,  c='tab:red')
    plt.xlabel('num_train')
    plt.ylabel('MSE')
    plt.yscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.title(os.path.basename(save_name))
    plt.savefig(save_name + 'test_loss_overlay.png')

###################################
# # Plotting VAR correlation plot #
###################################
def plot_coref(cur_folder, num_trails, save_name, coef_loader, coef_name):
    f = plt.figure(figsize=[8, 4])
    var_folder = cur_folder.replace('MSE','VAR')
    coeff = coef_loader(var_folder, num_trails)
    for i in range(num_trails):
        plt.plot(coeff[i], 'x', alpha=0.2)
    plt.ylim([0, 1])
    plt.xlabel('epoch')
    plt.ylabel('correlation')
    plt.savefig(save_name + coef_name + '_cor.png')

#######################################################
# Plot the Data efficiency plot here#
#######################################################
def Data_efficiency_plot(x_label, MSE_test_mse_mat, Random_test_mse_mat, save_name, VAR_test_mse_mat, NA_test_mse_mat, plot_var=True,
                    mse_upper_bound=0.1, num_points=20, y_lim=None, random_folder_provide=None,NAMD_test_mse_mat=None):
    """
    The is the data efficiency plot function.
    The X axis is the MSE level that it can get
    The y axis is the number of training data it needs to get to that position
    """

    f = plt.figure(figsize=[8, 4])
    if VAR_test_mse_mat is None:
        VAR_test_mse_mat = Random_test_mse_mat
    
    VAR_efficiency_mat, MSE_efficiency_mat,  = np.zeros([num_trails,num_trails, num_points]), np.zeros([num_trails,num_trails, num_points])
    NA_efficiency_mat, NAMD_efficiency_mat= np.zeros([num_trails,num_trails, num_points]), np.zeros([num_trails,num_trails, num_points])
    # First of all, get the x axis points
    for i in range(num_trails):
        for k in range(num_trails):
            # Get the each of the MSE for single trail
            VAR_mse = VAR_test_mse_mat[:, i]
            RD_mse = Random_test_mse_mat[:, k]          # This is the reference point, which is governed by k
            MSE_mse = MSE_test_mse_mat[:, i]
            NA_mse = NA_test_mse_mat[:, i]
            if plot_namd:
                NAMD_mse = NAMD_test_mse_mat[:, i]
            # Get the MSE min and max
            MSE_max = np.min([np.max(VAR_mse), np.max(RD_mse), np.max(MSE_mse), np.max(NA_mse), mse_upper_bound])
            MSE_min = np.max([np.min(VAR_mse), np.min(RD_mse), np.min(MSE_mse), np.min(NA_mse)])
            # Get the X-axis
            MSE_list = np.linspace(np.log(MSE_min), np.log(MSE_max), num=num_points)
            # Get the Y-axis
            for j in range(len(MSE_list)):
                mse_cur = np.exp(MSE_list[j])
                num_train_RD = x_label[np.argmin(np.square(RD_mse - mse_cur))]
                num_train_MSE = x_label[np.argmin(np.square(MSE_mse - mse_cur))]
                num_train_VAR = x_label[np.argmin(np.square(VAR_mse - mse_cur))]
                num_train_NA = x_label[np.argmin(np.square(NA_mse - mse_cur))]
                VAR_efficiency_mat[i, k, j] = num_train_VAR/num_train_RD
                MSE_efficiency_mat[i, k, j] = num_train_MSE/num_train_RD
                NA_efficiency_mat[i, k, j] = num_train_NA/num_train_RD
                if plot_namd:
                    num_train_NAMD = x_label[np.argmin(np.square(NAMD_mse - mse_cur))]
                    NAMD_efficiency_mat[i, k, j] = num_train_NAMD/num_train_RD
    for i in range(num_trails):
        for k in range(num_trails):
            # # Start the plotting
            # if i == 0:
            #     plt.plot(np.exp(MSE_list), MSE_efficiency_mat[i, :], '--x',c='tab:blue', label='MSE efficiency',alpha=alpha)
            #     if plot_var:
            #         plt.plot(np.exp(MSE_list), VAR_efficiency_mat[i, :], '--x',c='tab:green', label='VAR efficiency',alpha=alpha)
            #     if plot_na:
            #         plt.plot(np.exp(MSE_list), NA_efficiency_mat[i, :], '--x',c='tab:purple', label='NA efficiency',alpha=alpha)
            # else:
            plt.plot(np.exp(MSE_list), MSE_efficiency_mat[i,k, :], '--x', c='tab:blue',alpha=alpha)
            if plot_var:
                plt.plot(np.exp(MSE_list), VAR_efficiency_mat[i, k, :], '--x',c='tab:green',alpha=alpha)
            if plot_na:
                plt.plot(np.exp(MSE_list), NA_efficiency_mat[i, k, :], '--x',c='m',alpha=alpha)
            if plot_namd:
                plt.plot(np.exp(MSE_list), NAMD_efficiency_mat[i, k, :], '--x',c='tab:red',alpha=alpha)
    #print(len(MSE_efficiency_mat))
    #print(np.shape(np.array(MSE_efficiency_mat)))
    plt.plot(np.exp(MSE_list), np.mean(np.mean(MSE_efficiency_mat, axis=0),axis=0), '-x', c='b', label='MSE mean')
    if plot_var:
        plt.plot(np.exp(MSE_list), np.mean(np.mean(VAR_efficiency_mat, axis=0),axis=0), '-x', c='g', label='VAR mean')
    if plot_na:
        plt.plot(np.exp(MSE_list), np.mean(np.mean(NA_efficiency_mat, axis=0),axis=0), '-x', c='m', label='NA mean')
    if plot_namd:
        plt.plot(np.exp(MSE_list), np.mean(np.mean(NAMD_efficiency_mat, axis=0),axis=0), '-x', c='r', label='NAMD mean')
    plt.xlabel('MSE')
    plt.ylabel('Efficiency')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    if y_lim:
        plt.ylim([0, y_lim])
    plt.grid()
    plt.legend()
    plt.savefig(save_name + '_data_efficiency.png')
    
def get_dx_x0(name):
    """
    Get the dx and x0 from a name
    """
    dx = int(name.split('_dx_')[-1].split('_')[0])
    x0 = int(name.split('_x0_')[-1].split('_')[0])
    print('for {}, dx = {}, x0 = {}'.format(name, dx, x0))
    return dx, x0

def get_x_label_from_folder_name(random_folder_provide):
    for file in os.listdir(random_folder_provide):
        cur_folder = os.path.join(random_folder_provide, file)
        if not os.path.isdir(cur_folder):
            continue
        dx, x0 = get_dx_x0(cur_folder)
        mse = np.load(os.path.join(cur_folder, 'test_mse'))
        print('len of mse in get x label', np.len(mse))
        x_label = np.array(range(len(mse))) * dx + x0
        print('x_label = ', x_label)
        return x_label
