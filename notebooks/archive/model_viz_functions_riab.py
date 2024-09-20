from dsm.state import FittedValueTrainState

import jax
import jax.numpy as jnp

import einops
import numpy as np


#######################################
# plotting generated samples
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
cmap1 = plt.get_cmap("viridis").reversed()  # plt.get_cmap('Dark2')
cmap = plt.get_cmap('tab10')

#####################
import decoder_PC2d
#####################


def plot_samples(PCs, dataset_observation, samples,source, make_dataset=False,atom='all',method='dropoutNet'):

    if make_dataset:
        # dataset_positions = decoder_PC2d.simple_decode_position(dataset.observation,env_coords,pc_full_env,plot=False)
        dataset_positions = decoder_PC2d.decode_position(PCs,dataset_observation,
                                                         plot=False,method=method)
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3 * 2, 3))
        # # Left scatter plot
        # # converts Cartesian coordinates to polar coordinates (thetas) and extracts velocities
        xs = dataset_positions[:, 0]
        ys = dataset_positions[:, -1]
        colors = range(len(xs))
        axs[0].scatter(xs, ys, alpha=0.1, c=colors, cmap=cmap1)
        # axs[0].scatter(xs[0], ys[0], color='k',s=20)
        axs[0].scatter(xs[-1], ys[-1], color='red',s=50)
        # axs.plot(xs, ys, color='blue',alpha=0.2)
        # axs.set_xlim([0,1])
        # axs.set_ylim([0,1])
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    # Plot atom scatter & kde
    # plots generated samples - each colour represents atom?
    # cmap = plt.get_cmap(cmap1)  #"Dark2"# pyright: ignore
    if isinstance(atom, int):
        atom_samples = samples[atom]
        # atom_samples_xy = decoder_PC2d.simple_decode_position(atom_samples,env_coords,pc_full_env,plot=False)
        atom_samples_xy = decoder_PC2d.decode_position(PCs,atom_samples,
                                                       plot=False,method=method)
        # print('atom_samples_xy',atom_samples_xy)
        xs = atom_samples_xy[:, 0]
        ys = atom_samples_xy[:, -1]
        if make_dataset:
            axs[1].scatter(xs, ys, color=cmap(atom), s=50, alpha=0.25)
        else:
            plt.scatter(xs, ys, color=cmap(atom), s=50, alpha=0.25)
        plt.title(f"Generated samples of atom {atom}")
    elif atom=='all': #elif isinstance(atom, np.ndarray):
        atom_idx_values = np.arange(samples.shape[0])
        norm = Normalize(vmin=min(atom_idx_values), vmax=max(atom_idx_values))
        sm = ScalarMappable(cmap=cmap, norm=norm)

        for atom_idx in atom_idx_values:
            atoms_samples = samples[atom_idx] #.reshape(1,-1)
            # atom_samples_xy = decoder_PC2d.simple_decode_position(atom_samples,env_coords,pc_full_env,plot=False)
            # atoms_samples_xy = decoder_PC2d.decode_position(PCs,atoms_samples,plot=False,method=method)
            atoms_samples_xy, positions_std = decoder_PC2d.decode_position(PCs,atoms_samples,
                                                                           plot=False,method=method,return_std=True)
            assert atoms_samples_xy.shape[1] == 2 #atoms_samples_xy.shape[0] == num_samples and
            xs = atoms_samples_xy[:, 0]
            ys = atoms_samples_xy[:, -1]
            if (xs < 0).any() or (xs > 1).any() or (ys < 0).any() or (ys > 1).any():
                print(f"Warning: some samples are outside the bounds for atom {atom_idx}")
            
            maxc1, maxc2 = np.max(positions_std[:, 0]), np.max(positions_std[:, 1])
            if maxc1 > 0.1 and maxc2 > 0.1:
                print(f'Max SD of samples decoded with {method}: {maxc1, maxc2} ; Atom {atom_idx}')

            # colors = np.arange(len(xs))
            # print(cmap(i))
            if make_dataset:
                # axs[1].scatter(xs, ys,c=colors,s=40, cmap=cmap, alpha=0.25)
                axs[1].scatter(xs, ys,color=cmap(norm(atom_idx)),s=40, alpha=0.25)
            else:
                # plt.scatter(xs, ys,c=colors,s=40, cmap=cmap, alpha=0.25)
                plt.scatter(xs, ys, color=cmap(norm(atom_idx)),s=40, alpha=0.25)
        ax = axs[1] if make_dataset else plt.gca()
        plt.colorbar(sm, ax=plt.gca(), label='atom_idx')
        plt.title("Decoded sample positions of all atoms")
    # Plot source state
    source_xy = decoder_PC2d.decode_position(PCs,source.reshape(1,-1),plot=False,method=method)[0]
    print('decoded source: ',source_xy)
    if make_dataset:
        for ax in axs:
            ax.scatter(source_xy[0], source_xy[-1], marker="x", s=32, alpha=0.8, color="red")
    else:
        plt.scatter(source_xy[0], source_xy[-1], marker="x", s=64, alpha=0.8, color="red")

    # set bounds
    if make_dataset:
        for ax in axs:
            # ax.set_ylim(-8.5, 8.5)
            ax.set_aspect("auto")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            # ax.set_xlim([-0.5, 1.5])
            # ax.set_ylim([-0.5, 1.5])
    else:
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        # plt.ylim(-0.5, 1.5)
        # plt.xlim(-0.5, 1.5)
    # image = plotting_utils.fig_to_ndarray(fig)
    plt.show(fig)
########################################
import model_viz_loaders
def plot_predictions_atoms(config, latent_rng_seed, model_generator, source_states, 
                              PCs,atoms_sel = 'all', method='dropoutNet',title=None):
    layers = ['Dense_3',]  # predicted_pcs same as 3rd layer activations
    zs = jax.random.normal(jax.random.PRNGKey(latent_rng_seed), (config.latent_dims,)) 
    if atoms_sel == 'all':
        atoms_sel = range(config.num_outer)
    else:
        assert isinstance(atoms_sel, list)
    preds_all_atoms_list = []
    for atom_idx in atoms_sel:
        # print(atom_idx)
        atom_params = model_viz_loaders.extract_params_ith_atom(model_generator, atom_idx, config.num_outer)
        predicted_pcs_allsources = compute_activation_layer_all_sources(model_generator, atom_params, source_states,zs, layers)
        assert predicted_pcs_allsources.shape == source_states.shape
        positions_pred  = decoder_PC2d.decode_position(PCs, predicted_pcs_allsources, method=method)
        preds_all_atoms_list.append(positions_pred)

    n = len(preds_all_atoms_list)
    cols = 5 
    rows = n // cols 
    rows += n % cols
    position = range(1,n + 1)
    fig = plt.figure() #figsize=(10, 10)
    for k, positions_pred in enumerate(preds_all_atoms_list): 
        ax = fig.add_subplot(rows, cols, position[k])
        ax.set_aspect('equal')
        xs, ys = positions_pred[:,0], positions_pred[:,1]
        colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1
        
        ax.scatter(xs, ys, alpha=0.2, c=colors, cmap=cmap1)  # Use the numbers as colors
        ax.scatter(xs[0], ys[0], color='k',s=20)
        ax.scatter(xs[-1], ys[-1], color='red',s=25)
        # ax.plot(xs, ys, color='blue',alpha=0.2)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        # ax.set_xlim([-0.5,1.5])
        # ax.set_ylim([-0.5,1.5])
        # plt.xticks([]) 
        # plt.yticks([]) 
    if title is not None:
        fig.suptitle(title)
    plt.show() 



##########################################


NUM_STATE_DIM_CELLS = 50

################################################################################################################

def predict_with_all_intermediates(model_generator,params, x): 
    # mlp_model = MLP(num_layers=3, num_hidden_units=8, num_outputs=4)
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = NUM_STATE_DIM_CELLS
    outputs, intermediates = mlp_model_atom.apply({"params": params}, x, capture_intermediates=True)
    return outputs, intermediates

def predict_layer(model_generator, params, x, layers):  #self, 
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = NUM_STATE_DIM_CELLS
    # filter specifically for only the layers like Dense_0 and Dense_2 
    filter_fn = lambda mdl, method_name: isinstance(mdl.name, str) and (mdl.name in layers )
    return mlp_model_atom.apply({"params": params}, x, capture_intermediates=filter_fn)

# xs_atom = xs[0][2]
# layers = {'Dense_3',} #, 'Dense_2'
# preds, feats = predict_layer(atom_params, xs_atom,layers)
# print(preds)
# print(feats)
# preds, intermediates = predict_with_all_intermediates(atom_params, xs_atom)
# # print(intermediates['intermediates'].keys())  # This will contain the intermediate outputs
# # feats # intermediate c in Model is stored and isn't filtered out by the filter function
################################################################################################################

def compute_activation_layer_1source(model_generator, atom_params, source_state,zs, layers):
    # zs = jax.random.normal(rng, (num_latent_dims,))
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
    activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
    # print('activations: ',activations_layer)
    return activations_layer

def compute_activation_layer_all_sources(model_generator, atom_params, sources_all,zs, layers):
    ''' extracts the source as 50 place cells in 1 of the 10k states'''
    activations_all = []
    for source_state in sources_all:
        activations_all.append(compute_activation_layer_1source(model_generator, atom_params, 
                                                                source_state,zs, layers))
    return np.array(activations_all)

#####################################

def compute_activation_allintermediates_1source(model_generator, atom_params, source_state, zs, layers):
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    _, intermediates = predict_with_all_intermediates(model_generator, atom_params, xs_atom)
    
    activations_layers = {}
    for layer in layers:
        activations_layers[layer] = intermediates['intermediates'][layer]['__call__'][0]
    return activations_layers


def compute_activation_allintermediates_all_sources(model_generator, atom_params, sources_all, zs, layers):
    activations_all = {layer: [] for layer in layers}
    
    for source_state in sources_all:
        activations_source = compute_activation_allintermediates_1source(model_generator, atom_params, 
                                                                         source_state, zs, layers)
        for layer in layers:
            activations_all[layer].append(activations_source[layer])
    # Convert lists to numpy arrays
    for layer in layers:
        activations_all[layer] = np.array(activations_all[layer])
    
    return activations_all
################################################

import matplotlib.ticker as ticker
cmap_ratemap = "inferno"            
def plot_neuron_activations(activations_layer, neuron_idx, figlabel, xpos,ypos, ax_main=None, normalize=False):
    
    if isinstance(neuron_idx, str) and neuron_idx == 'all':
        neuron_idx = np.arange(activations_layer.shape[-1])
    
    # plot for single neuron
    if isinstance(neuron_idx, int): #if not isinstance(neuron_idx,np.ndarray): 
        print(activations_layer.shape)
        activations_selected = activations_layer[:, neuron_idx]
        # outputs_reshaped = activations_selected.reshape(X.shape)
        if normalize:
            activations_selected = (activations_selected - np.amin(activations_selected)) / (np.amax(activations_selected) - np.amin(activations_selected))

        if ax_main==None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
        else:
            ax = ax_main
        activations_2d = activations_selected.reshape(len(ypos), len(xpos))
        scatter = ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
        ax.set_aspect('equal')
        # scatter = ax.scatter(angles_flat, Z_flat, c=activations_selected) #, cmap='viridis'
        # Add a color bar to indicate the output values

        ax.set_title(f'{figlabel}')
        if ax_main==None:
            fig.colorbar(scatter, ax=ax, label='Neuron Activity')
            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

    elif isinstance(neuron_idx,np.ndarray) or (isinstance(neuron_idx, str) and neuron_idx == 'avg'):
        # plot for multiple neurons in same layer
        fig = plt.figure()
        if isinstance(neuron_idx,np.ndarray):
            Tot = len(neuron_idx)
            if Tot % 10 == 0:
                Cols = 10
            else:
                Cols = 8 
            if Tot < Cols:
                Cols = Tot
        else:
            Tot = len(activations_layer)
            Cols = 5
        Rows = Tot // Cols 
        if Tot % Cols != 0:
            Rows += 1
        Position = range(1,Tot + 1)
        # print(activations_layer.shape, Tot)
        if isinstance(neuron_idx,np.ndarray):
            for k,neuron_i in enumerate(neuron_idx):
                activations_selected = activations_layer[:, neuron_i]
                # print(activations_selected.shape)            
                ax = fig.add_subplot(Rows,Cols,Position[k])
                activations_2d = activations_selected.reshape(len(ypos), len(xpos))
                ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
                ax.set_aspect('equal')
                # scatter = ax.scatter(angles_flat, Z_flat, c=activations_selected)

                # ax = fig.add_subplot(Rows,Cols,Position[k], projection='3d')
                # scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=activations_selected)
                # # plt.yticks([]) 
                # # plt.xticks([]) 
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                # ax.zaxis.set_major_locator(ticker.NullLocator())
            # fig.suptitle(f'Neuron Activations of layer {label_layer}')
        elif isinstance(neuron_idx, str) and neuron_idx == 'avg':
            for atom_i in range(len(activations_layer)):
                activations_avg = np.mean(activations_layer[atom_i], axis=1)
                ax = fig.add_subplot(Rows,Cols,Position[atom_i])
                activations_2d = activations_avg.reshape(len(ypos), len(xpos))
                ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
                ax.set_aspect('equal')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.set_title(f'Atom: {atom_i}')
            fig.suptitle(f'{figlabel}')
    else:
        raise ValueError('neuron_idx should be an integer or numpy array or neuron_idx == avg')


def plot_neuron_ratemaps_from_activationlayerlist(activations_layer_list, figlabel, neuron_idx, xpos, ypos, normalize=True):
    n = len(activations_layer_list)
    cols = 5 
    rows = n // cols 
    rows += n % cols
    position = range(1,n + 1)
    fig = plt.figure() #figsize=(10, 10)
    for k, activations_layer in enumerate(activations_layer_list): 
        ax = fig.add_subplot(rows, cols, position[k])
        ax.set_aspect('equal')
        label = f'{figlabel}: {k}'
        plot_neuron_activations(activations_layer,neuron_idx,label,xpos,ypos, ax_main=ax, normalize=normalize)
        plt.xticks([]) 
        plt.yticks([]) 
    plt.show()

########################################

def euclidean_distance(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2) ** 2))

def cosine_similarity(arr1, arr2):
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

def compute_distance_similarity_with_trueratemap(true_ratemap_all,  activations_layer_1atom_allckpts, 
                                                 ckpt_step_nums,neuron_list,metric='euclidean'):
    num_ckpts = len(activations_layer_1atom_allckpts)
    for neuron_idx in neuron_list:
        print('Neuron: ',neuron_idx)
        for i in range(num_ckpts):
            act_neuron1  = activations_layer_1atom_allckpts[i][:, neuron_idx] 
            true_neuron_ratemap = true_ratemap_all[neuron_idx]
            if metric == 'euclidean':
                score = euclidean_distance(act_neuron1, true_neuron_ratemap)
            elif metric == 'cosine':
                score = cosine_similarity(act_neuron1, true_neuron_ratemap)
            print(f'Ckpt Step: {ckpt_step_nums[i]}: {metric} score:', score)
            
def compute_distance_similarity(activations_layer_1atom_allckpts,ckpt_step_nums,neuron_list,metric='euclidean'):
    num_ckpts = len(activations_layer_1atom_allckpts)
    for neuron_idx in neuron_list:
        print('Neuron: ',neuron_idx)
        for i in range(num_ckpts-1):
            activations_layer1, activations_layer2 = \
                    activations_layer_1atom_allckpts[i], activations_layer_1atom_allckpts[i+1]
            act_neuron1, act_neuron2 = \
                    activations_layer1[:, neuron_idx], activations_layer2[:, neuron_idx]
            if metric == 'euclidean':
                score = euclidean_distance(act_neuron1, act_neuron2)
            elif metric == 'cosine':
                score = cosine_similarity(act_neuron1, act_neuron2)
            print(f'Ckpt Step: {ckpt_step_nums[i]} - {ckpt_step_nums[i+1]}: \
                  {metric} score:', score)
########################################

def multiple_samples_compute_activation_layer_1source(model_generator, atom_params, source_state, layers, 
                                                      num_samples, num_latent_dims, random_seed = 0 ):
    context = einops.repeat(source_state, "s -> i s", i=num_samples)
    zs = jax.random.normal(jax.random.PRNGKey(random_seed), (num_samples, num_latent_dims))
    xs_all = jnp.concatenate((zs, context), axis=-1)

    all_samples_activations= []
    for xs_atom in xs_all:
        preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
        activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
        all_samples_activations.append(activations_layer)
    all_samples_activations = np.array(all_samples_activations)

    return all_samples_activations

def show_allsamples_compute_activation_layer_all_sources(model_generator, atom_params, sources_all, layers,num_samples,num_latent_dims):
    activations_all = []
    for source_state in sources_all:
        activations_all.append(multiple_samples_compute_activation_layer_1source(model_generator, atom_params, 
                                                                                 source_state, layers,num_samples,num_latent_dims))
    return np.array(activations_all)

########################################

def plot_neuron_activations_samples(activations_layer, neuron_idx,label,xpos, ypos):
    # plot for single neuron
    # plt for each source - several samples
    fig = plt.figure()
    Tot = activations_layer.shape[1]
    Cols = 8
    if Tot < Cols:
        Cols = Tot
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)

    activations_neuron = activations_layer[:,:, neuron_idx]
    # outputs_reshaped = activations_selected.reshape(X.shape)

    for k in range(Tot):
        activations_selected = activations_neuron[:, k]
        ax = fig.add_subplot(Rows,Cols,Position[k])
        ax.scatter(xpos, ypos, c=activations_selected)
    
    plt.title(label)



# plot_module.plot_samples(source, state, rng, config=config)
#             for (_, source) in zip(*plot_module.source_states())
# env_coords_small = Ag.Environment.discretise_environment(dx=0.3) # dx=Ag.environment.scale/10
# env_coords_small = env_coords_small.reshape(-1, env_coords_small.shape[-1])

# env_coords_small.shape


# # buildable from configs file
# fiddle.extensions.jax.enable()
# FLAGS = flags.FLAGS
# # flags.DEFINE_string('fdl_config', 'base', 'The Fiddle configuration to use.')
# if not FLAGS.fdl_config and not FLAGS.fdl_config_file:
#         FLAGS.fdl_config = 'base'
# buildable = fdl_flags.create_buildable_from_flags(configs)
# config: configs.Config = fdl.build(buildable)
# num_outer=config.num_outer # Number of model atoms   
# num_latent_dims= config.latent_dims # Dimension of input noise 
# print('num_samples', num_samples,' num_outer(no of atoms):', num_outer, ' num_latent_dims ',num_latent_dims)



# from dsm import configs_PCs_teleport as configs #The decorrelation timescale can be also be controlled. 





def resultant_vector(alpha_bins, nanrobust, axis=None, w=None, d=0):
    """
    Calculate the resultant vector of a given set of angles.

    Parameters:
    alpha_bins (ndarray): Array of angles.
    nanrobust (bool): Flag indicating whether to handle NaN values robustly.
    axis (int or tuple of ints, optional): Axis or axes along which the sum is computed. Default is None.
    w (ndarray, optional): Array of weights. Default is None.
    d (float, optional): Known spacing between angles. Default is 0.
    """

    if w is None:
        w = np.ones(alpha_bins.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha_bins)
    if nanrobust:
        r = np.nansum(t, axis=axis)
        w[np.isnan(t)] = 0
    else:
        r = np.sum(t, axis=axis)

    r_angle = np.angle(r)

    # obtain length
    r = np.abs(r) / np.sum(w, axis=axis)

    # for data with known spacing, apply correction factor to correct
    # for bias in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d != 0:
        r *= d / 2 / np.sin(d / 2)

    return r, r_angle
# where alpha_bins is a list of angles (one for each bin), 
# w is the normalized polar map of a hidden unit (list of 20 values) 
# and d is just 18 (degrees) in this case. 
# the function returns the value and the angle of the resultant vector.

def plot_theta_():
    # where thetas_ticks is a list of the value of the mid point of each bin, 
    # data_latent_polar_n is a list of the value for each bin (i.e. the average activation per bin of the hidden unit you are plotting), 
    # and r_angle is the angle of the resulting Rayleigh vector for the polar map of that hidden unit.
    ax = plt.subplots(
        1, 1, subplot_kw={'projection': 'polar'},
        figsize=(5,5)
    )
    ax.plot(
        np.append(self.thetas_ticks, self.thetas_ticks[0]), # we want to close the circle
        np.append(data_latent_polar_n, data_latent_polar_n[0]), # we want to close the circle
        lw=3, c='blue',
        # marker='o', ms=5, mfc='red'
    )
    ax.vlines(
        r_angle,
        0, np.max(data_latent_polar_n),
        colors='red', lw=2
    )
    ax.set_xticklabels([]) # remove degrees indication
    ax.set_rticks([]) # remove intensity indication
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N') # move 0 to the north
    ax.grid(True)
    

    # I classify a hidden unit as HD if its normalized (divided by the number of bins) polar map’s KL-divergence against 
    # a uniform distribution is higher than 0.15 or if its normalized polar map’s resultant vector is longer than 0.3 
    # (these are quite traditional thresholds used in neuroscience papers, but you should discuss with Caswell how to best set them). 
    # I calculate the KL divergence like so:
    def kl_divergence(p, q, eps=1e-10):
        # clip values to avoid log(0)
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        return np.sum(p * np.log(p / q))
    # where p is the polar map of a hidden unit (list of 20 values in my case) and q is just the uniform distribution (list of ones over 20). 
    # I calculate the resultant vector like so:


###############################################

# env_coords_point05 = Ag.Environment.discretise_environment(dx=0.05) # dx=Ag.environment.scale/10 , dx=0.01
# # dx=0.05 , (400, 2)
# #dx = 0.2, (25,2)
# env_coords_point05 = env_coords_point05.reshape(-1, env_coords_point05.shape[-1])
# # pc_full_env = PCs.get_state(evaluate_at="all").T # N of 10000 values corresponding to ag.Environment.flattened_discrete_coords # len 10000
# source_PCs = PCs.get_state(evaluate_at=None, pos=env_coords_point05).T
# source_PCs[:,0]=0
# print('CHANGE BASIS VECTOR: source[:,0]=0')

# # source_PCs = source_states_env #[::2]
# latent_rng_seed = 2 
# positions_source  = decoder_PC2d.decode_position(PCs, source_PCs , plot=True, method='dropoutNet')
# modelviz_utils.plot_predictions_atoms(config, latent_rng_seed, state.generator, source_PCs, 
#                               PCs, method='dropoutNet')

