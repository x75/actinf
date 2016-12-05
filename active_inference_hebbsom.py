# test conditional inference using intermodal maps built with soms and hebbian interconnections
#
# Oswald Berthold, 09/2016
#
# candidates:
#  - pyERA: proved a) too laborious and b) too slow for some reason
#  - neupy: couldn't quickly find an online learning mode
#  - kohonen: thx lmjohns3, kohonen lib is doing the base SOMs, hebbian is custom


import sys, argparse
import cPickle, os
import numpy as np
import pylab as pl

dim_e = 2 # exteroceptive dim
dim_p = 3 # proprioceptive dim

def main_era(args):
    from pyERA.som import Som
    from pyERA.hebbian import HebbianNetwork
    from pyERA.utils import ExponentialDecay

    som_e = Som(matrix_size = 3, input_size = dim_e, low = -0.1, high = 0.1)
    som_p = Som(matrix_size = 3, input_size = dim_p, low = -0.01, high = 0.01)

    hebb_e2p = HebbianNetwork("e2p")
    hebb_e2p.add_node("som_e", (3, 3))
    hebb_e2p.add_node("som_p", (3, 3))
    hebb_e2p.add_connection(0, 1)
    hebb_e2p.add_connection(1, 0)
    hebb_e2p.print_info()
    
    print "som_e weight_matrix", som_e.return_weights_matrix()
    print "som_p weight_matrix", som_p.return_weights_matrix()

    learning_rate_e = ExponentialDecay(starter_value=0.4, decay_step=50, decay_rate=0.9, staircase=True)
    radius_e = ExponentialDecay(starter_value=np.rint(som_e._matrix_size/3), decay_step=80, decay_rate=0.90, staircase=True)
    
    learning_rate_p = ExponentialDecay(starter_value=0.4, decay_step=50, decay_rate=0.9, staircase=True)
    radius_p = ExponentialDecay(starter_value=np.rint(som_p._matrix_size/3), decay_step=80, decay_rate=0.90, staircase=True)

    # data
    EP = np.load("data/simplearm_n1000/EP_1000.npy")
    
    for i in range(EP.shape[0]):
        # get data item
        E = EP[i,:dim_e]
        P = EP[i,dim_e:]

        # get learning params
        learning_rate_e_ = learning_rate_e.return_decayed_value(global_step=i)
        radius_e_ = radius_e.return_decayed_value(global_step=i)
        learning_rate_p_ = learning_rate_p.return_decayed_value(global_step=i)
        radius_p_ = radius_p.return_decayed_value(global_step=i)

        # activate and fit
        som_e_activation = som_e.return_activation_matrix(E)
        som_e_bmu_value = som_e.return_BMU_weights(E)
        bmu_index = som_e.return_BMU_index(E)
        bmu_weights = som_e.get_unit_weights(bmu_index[0], bmu_index[1])
        bmu_neighborhood_list = som_e.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius_e_)
        som_e.training_single_step(E, units_list=bmu_neighborhood_list, learning_rate = learning_rate_e_, radius=radius_e_, weighted_distance = False)

        som_p_activation = som_p.return_activation_matrix(P)
        bmu_index = som_p.return_BMU_index(P)
        bmu_weights = som_p.get_unit_weights(bmu_index[0], bmu_index[1])
        bmu_neighborhood_list = som_p.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius_p_)
        som_p.training_single_step(P, units_list=bmu_neighborhood_list, learning_rate = learning_rate_p_, radius=radius_p_, weighted_distance = False)

        # hebbian prediction
        print("E: " + str(E) + "; E winner: " + str(som_e_bmu_value) + "\n")
        hebb_e2p.set_node_activations(0, som_e_activation)
        som_p_hebbian_matrix = hebb_e2p.compute_node_activations(1, set_node_matrix=False)
        print "som_e_activation", som_e_activation
        print "diff native/hebbian", som_p_activation, som_p_hebbian_matrix
        max_row, max_col = np.unravel_index(som_p_hebbian_matrix.argmax(), som_p_hebbian_matrix.shape)
        P_weights = som_p_bmu_weights = som_p.get_unit_weights(max_row, max_col)
        P_ = P_weights
        print "P_", P_, "P", P

        # hebbian training / association
        hebb_e2p.set_node_activations(0, som_e_activation)
        hebb_e2p.set_node_activations(1, som_p_activation)
        # 3 - Positive Learning!
        hebb_e2p.learning(learning_rate=0.1, rule="hebb")
        

    som_e_weights = som_e.return_weights_matrix()
    zero_chan = np.zeros((som_e_weights.shape[0], som_e_weights.shape[1], 1))
    print "som_e_weights", som_e_weights.shape, som_e_weights
    img_e = np.rint(np.concatenate((som_e_weights*255, zero_chan), axis=2))

    som_p_weights = som_p.return_weights_matrix()
    zero_chan = np.zeros((som_p_weights.shape[0], som_p_weights.shape[1], 1))
    print "som_p_weights", som_p_weights.shape, som_p_weights
    img_p = np.rint(som_p_weights*255)
    
    print img_e.shape, img_p.shape
    pl.subplot(211)
    pl.axis("off")
    pl.imshow(img_e, interpolation="none")
    pl.subplot(212)
    pl.axis("off")
    pl.imshow(img_p, interpolation="none")
    pl.show()
    
        
def main_neupy(args):
    import matplotlib.pyplot as plt

    from neupy import algorithms, environment


    environment.reproducible()
    plt.style.use('ggplot')    

    
    # data
    EP = np.load("data/simplearm_n3000/EP.npy")

    input_data = EP[:,:2]
    
    sofmnet = algorithms.SOFM(
        n_inputs=2,
        n_outputs=20,

        step=0.01,
        show_epoch=10,
        shuffle_data=False,
        verbose=True,

        learning_radius=2,
        features_grid=(20, 1),
        )

    plt.plot(input_data.T[0:1, :], input_data.T[1:2, :], 'kx', alpha=0.5)
    sofmnet.train(input_data, epochs=100)

    print("> Start plotting")
    plt.xlim(-1, 1.2)
    plt.ylim(-1, 1.2)

    plt.plot(sofmnet.weight[0:1, :], sofmnet.weight[1:2, :], 'bo', markersize=8, linewidth=5)
    plt.show()

    for data in input_data:
        print(sofmnet.predict(np.reshape(data, (2, 1)).T))

def main_kohonen(args):
    """testing the map from extero to proprio sensor subspaces

    it is general in that if the map is underdetermined (dim_e < dim_p),
    the map can cope with multiple solutions and samples from the joint density

    in-situ Hebbian linked SOMs (Miikkulainen) using lmjohns3's kohonen python lib"""
    
    from kohonen.kohonen import Map, Parameters, ExponentialTimeseries, ConstantTimeseries
    from kohonen.kohonen import Gas, GrowingGas, GrowingGasParameters, Filter
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D

    # extract datadir from datfile
    datadir = "/".join(args.datafile.split("/")[:-1])
    # load offline data from active inference experiment
    # EP = np.load("EP_1000.npy") # [:200]
    # EP = np.load("EP.npy") # [:500]
    # EP = np.load("EP.npy")[500:600]
    EP = np.load(args.datafile)

    som_e_save = datadir + "/" + "som_filter_e.pkl"
    som_p_save = datadir + "/" + "som_filter_p.pkl"
    som_e2p_link_save = datadir + "/" + "som_e2p_link.npy"
    
    # learning rate proxy
    ET = ExponentialTimeseries

    # som argument dict
    def kwargs(shape=(10, 10), z=0.001, dimension=2, lr_init = 1.0, neighborhood_size = 1):
        return dict(dimension=dimension,
                    shape=shape,
                    neighborhood_size = neighborhood_size,
                    learning_rate=ET(-1e-4, lr_init, 0.01),
                    noise_variance=z)
    
    mapsize = 10
    # FIXME: make neighborhood_size decrease with time
    
    # SOM exteroceptive stimuli 2D input
    kw_e = kwargs(shape = (mapsize, mapsize), dimension = dim_e, lr_init = 0.5, neighborhood_size = 0.6)
    som_e = Map(Parameters(**kw_e))

    # SOM proprioceptive stimuli 3D input
    kw_p = kwargs(shape = (int(mapsize * 1.5), int(mapsize * 1.5)), dimension = dim_p, lr_init = 0.5, neighborhood_size = 0.7)
    som_p = Map(Parameters(**kw_p))
    
    # kw = kwargs((64, ))
    # som_e = Gas(Parameters(**kw))
    # kw = kwargs((64, ))
    # kw['growth_interval'] = 7
    # kw['max_connection_age'] = 17
    # som_e = GrowingGas(GrowingGasParameters(**kw))
    
    # kw_f_e = kwargs(shape = (mapsize, mapsize), dimension = 2, neighborhood_size = 0.75, lr_init = 1.0)
    # filter_e = Filter(Map(Parameters(**kw_f_e)))
    
    # create "filter" using existing SOM_e, filter computes activation on distance
    filter_e = Filter(som_e, history=lambda: 0.0)
    filter_e.reset()

    # kw_f_p = kwargs(shape = (mapsize * 3, mapsize * 3), dimension = 3, neighborhood_size = 0.5, lr_init = 0.1)
    # filter_p = Filter(Map(Parameters(**kw_f_p)), history=lambda: 0.01)
    
    # create "filter" using existing SOM_p, filter computes activation on distance
    filter_p = Filter(som_p, history=lambda: 0.0)
    filter_p.reset()

    # Hebbian links
    # hebblink_som    = np.random.uniform(-1e-4, 1e-4, (np.prod(som_e._shape), np.prod(som_p._shape)))
    # hebblink_filter = np.random.uniform(-1e-4, 1e-4, (np.prod(filter_e.map._shape), np.prod(filter_p.map._shape)))
    hebblink_som    = np.zeros((np.prod(som_e._shape), np.prod(som_p._shape)))
    hebblink_filter = np.zeros((np.prod(filter_e.map._shape), np.prod(filter_p.map._shape)))

    # # plot initial weights as images
    # pl.subplot(211)
    # pl.imshow(hebblink_som + 0.5, interpolation="none", norm=mcolors.NoNorm())
    # pl.colorbar()
    # pl.subplot(212)
    # pl.imshow(hebblink_filter + 0.5, interpolation="none", norm=mcolors.NoNorm())
    # pl.colorbar()
    # pl.show()

    fig = pl.figure()
    numepisodes_som = 10 # 30
    numepisodes_hebb = 10 # 30
    numsteps = EP.shape[0]
    
    ################################################################################
    # check for trained SOM
    if not os.path.exists(som_e_save) and \
        not os.path.exists(som_p_save):
    
        ################################################################################
        # train the SOMs on the input
        # som_e_barbase = np.linspace(-2, 2, np.prod(filter_e.map._shape))
        # som_p_barbase = np.linspace(-2, 2, np.prod(filter_p.map._shape))
        
        pl.ion()
        # pl.figure()
        # pl.show()
        for j in range(numepisodes_som):
            for i in range(numsteps):
                e = EP[i,:dim_e]
                p = EP[i,dim_e:]

                # don't learn twice
                # som_e.learn(e)
                # som_p.learn(p)
                filter_e.learn(e)
                filter_p.learn(p)
                # print np.argmin(som_e.distances(e)) # , som_e.distances(e)
    
                # on plot interval, update plot :)
                if i % 50 == 0:
                    ax = fig.add_subplot(121)
                    # pl.subplot(121)
                    ax.cla()
                    ax.plot(filter_e.map.neurons[:,:,0], filter_e.map.neurons[:,:,1], "ko", alpha=0.5, ms=10)
                    # data
                    ax.plot(e[0], e[1], "bx", alpha=0.7, ms=12)
                    # winner
                    # ax.plot(som_e.neurons[w_e[0],w_e[1],0], som_e.neurons[w_e[0],w_e[1],1], "ro", alpha=0.7, ms=12)
                    # ax.bar(som_e_barbase, filter_e.distances(e).flatten(), width=0.01, bottom = -2, alpha=0.1)
                    # ax.stem(filter_e.distances(e), "k", alpha=0.1)
                    ax.set_xlim((-2, 2))
                    ax.set_ylim((-2, 2))
                    ax.set_aspect(1)
                    ax.text(-1.0, 1.2, "%d / %d, eta = %f" % ((j * numsteps) + i, numepisodes_som * numsteps, som_e._learning_rate()))
    
                    ax = fig.add_subplot(122) #, projection='3d')
                    # pl.subplot(122)
                    ax.cla()
                    ax.plot(filter_p.map.neurons[:,:,0], filter_p.map.neurons[:,:,1], "ko", alpha=0.5, ms=10)
                    # ax.bar(som_p_barbase, filter_p.distances(p).flatten(), width=0.01, bottom = -2, alpha=0.1)
                    ax.set_xlim((-2, 2))
                    ax.set_ylim((-2, 2))
                    pl.gca().set_aspect(1)

                    pl.pause(0.01)
                    pl.draw()
                    # print som_e.neurons
                    
        ################################################################################
        # save SOMs
        # print "filter.map shapes", filter_e.map._shape, filter_p.map._shape
        cPickle.dump(filter_e.map, open(som_e_save, "wb"))
        cPickle.dump(filter_p.map, open(som_p_save, "wb"))
    
    else:
        print "loading existing SOMs"
        filter_e.map = cPickle.load(open(som_e_save, "rb"))
        filter_p.map = cPickle.load(open(som_p_save, "rb"))
        # filter_e.reset()
        # filter_p.reset()
        # print "filter.map shapes", filter_e.map._shape, filter_p.map._shape
        j = numepisodes_som - 1
        i = numsteps - 1

    # sys.exit()
        
    ################################################################################
    # plot final SOM configurations
    pl.ioff()
    # pl.subplot(121)
    ax = fig.add_subplot(121)
    ax.cla()
    ax.plot(filter_e.map.neurons[:,:,0], filter_e.map.neurons[:,:,1], "ko", alpha=0.5, ms=10)
    ax.plot(EP[:,0], EP[:,1], "r.", alpha=0.3, label="data_e")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_aspect(1)
    ax.text(-0.8, 0.7, "%d, eta = %f" % ((j * EP.shape[0]) + i, som_e._learning_rate()))
    ax.legend()
    # pl.subplot(224)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(filter_p.map.neurons[:,:,0], filter_p.map.neurons[:,:,1], filter_p.map.neurons[:,:,2], "ko", alpha=0.5)#, ms=10)
    ax.plot(EP[:,2], EP[:,3], EP[:,4], "r.", alpha=0.3, ms=5)
    #ax.set_xlim((-2, 2))
    #ax.set_ylim((-2, 2))
    #ax.set_zlim((-2, 2))
    pl.gca().set_aspect(1)
    pl.show()
    
    ################################################################################
    # fix the SOMs with learning rate constant 0
    CT = ConstantTimeseries
    filter_e.map.learning_rate = CT(0.0)
    filter_p.map.learning_rate = CT(0.0)
        
    ################################################################################
    # now train Hebbian associations on fixed SOM

    use_activity = True
    
    # Hebbian learning rate
    if use_activity:
        et = ExponentialTimeseries(-1e-4, 0.8, 0.001)
        # et = ConstantTimeseries(0.5)
    else:
        et = ConstantTimeseries(1e-5)
    e_shape = (np.prod(filter_e.map._shape), 1)
    p_shape = (np.prod(filter_p.map._shape), 1)

    z_err_coef = 0.99
    z_err_norm_ = 1
    Z_err_norm  = np.zeros((numepisodes_hebb*numsteps,1))
    Z_err_norm_ = np.zeros((numepisodes_hebb*numsteps,1))
    W_norm      = np.zeros((numepisodes_hebb*numsteps,1))

    if not os.path.exists(som_e2p_link_save):
        for j in range(numepisodes_hebb):
            # while 
            for i in range(numsteps):
                e = EP[i,:dim_e]
                p = EP[i,dim_e:]

                # just activate
                filter_e.learn(e)
                filter_p.learn(p)

                # fetch data induced activity
                if use_activity:
                    p_    = filter_p.activity.reshape(p_shape)
                else:
                    p_    = filter_p.distances(p).flatten().reshape(p_shape)
                
                # # get winner coord and flattened index
                # w_e_idx = som_e.winner(e)
                # # w_p_idx = som_p.winner(p)
                # w_e = som_e.flat_to_coords(w_e_idx)
                # # w_p = som_p.flat_to_coords(w_p_idx)
    
                # # get winner coord and flattened index
                # w_f_e_idx = filter_e.winner(e)
                # # w_f_p_idx = filter_p.winner(p)
                # w_f_e = filter_e.flat_to_coords(w_f_e_idx)
                # # w_f_p = filter_p.flat_to_coords(w_f_p_idx)
            
                # print som_e.neuron(w_e), som_p.neuron(w_p)

                # 
                # hebblink_som[w_e_idx,w_p_idx] += 0.01
            
                # hebblink_filter[w_f_e_idx,w_f_p_idx] += 0.01
            
                # print "filter_e.activity", filter_e.activity
                # print "filter_p.activity", filter_p.activity
                # print "np.outer(filter_e.activity.flatten(), filter_p.activity.flatten())", np.outer(filter_e.activity.flatten(), filter_p.activity.flatten()).shape
            
                # compute prediction for p using e activation and hebbian weights
                if use_activity:
                    p_bar = np.dot(hebblink_filter.T, filter_e.activity.reshape(e_shape))
                else:
                    p_bar = np.dot(hebblink_filter.T, filter_e.distances(e).flatten().reshape(e_shape))

                # inject activity prediction
                p_bar_sum = p_bar.sum()
                if p_bar_sum > 0:
                    p_bar_normed = p_bar / p_bar_sum
                else:
                    p_bar_normed = np.zeros(p_bar.shape)
            
                        
                # print np.linalg.norm(p_), np.linalg.norm(p_bar)
            
                # clip p_bar positive             
                # p_bar = np.clip(p_bar, 0, np.inf)
    
                # compute prediction error: data induced activity - prediction
                z_err = p_ - p_bar
                # z_err = p_bar - p_
                z_err_norm = np.linalg.norm(z_err, 2)
                if j == 0 and i == 0:
                    z_err_norm_ = z_err_norm
                else:
                    z_err_norm_ = z_err_coef * z_err_norm_ + (1 - z_err_coef) * z_err_norm
                w_norm = np.linalg.norm(hebblink_filter)

                logidx = (j*numsteps) + i
                Z_err_norm [logidx] = z_err_norm
                Z_err_norm_[logidx] = z_err_norm_
                W_norm     [logidx] = w_norm
            
                # z_err = p_bar - filter_p.activity.reshape(p_bar.shape)
                # print "p_bar.shape", p_bar.shape
                # print "filter_p.activity.flatten().shape", filter_p.activity.flatten().shape
                if i % 100 == 0:
                    print "iter %d/%d: z_err.shape = %s, |z_err| = %f, |W| = %f, |p_bar_normed| = %f" % (logidx, (numepisodes_hebb*numsteps), z_err.shape, z_err_norm_, w_norm, np.linalg.norm(p_bar_normed))
                    # print 
            
                # d_hebblink_filter = et() * np.outer(filter_e.activity.flatten(), filter_p.activity.flatten())
                if use_activity:
                    d_hebblink_filter = et() * np.outer(filter_e.activity.flatten(), z_err)
                else:
                    d_hebblink_filter = et() * np.outer(filter_e.distances(e), z_err)
                hebblink_filter += d_hebblink_filter
        np.save(som_e2p_link_save, hebblink_filter)

        ################################################################################
        # show final Hebbian weights
        pl.subplot(411)
        pl.imshow(hebblink_som + 0.5, interpolation="none", norm=mcolors.NoNorm())
        pl.colorbar()
        pl.subplot(412)
        # pl.imshow(hebblink_filter + 0.5, interpolation="none", norm=mcolors.NoNorm())
        pl.imshow(hebblink_filter, interpolation="none")
        pl.colorbar()
        pl.subplot(413)
        pl.plot(Z_err_norm, linewidth=0.5)
        pl.plot(Z_err_norm_)
        pl.gca().set_yscale("log")
        pl.subplot(414)
        pl.plot(W_norm)
        pl.show()
    else:
        print "Loading existing Hebbian link"
        hebblink_filter = np.load(som_e2p_link_save)
    


    # do prediction using activation propagation from extero map to proprio map via hebbian links
    from explauto import Environment
    environment = Environment.from_configuration('simple_arm', 'low_dimensional')
    environment.noise = 0.

    sampling_search_num = 100
    
    P_ = np.zeros((EP.shape[0], dim_p))    
    E_ = np.zeros((EP.shape[0], dim_e))
    e2p_w_p_weights = filter_p.neuron(filter_p.flat_to_coords(filter_p.sample(1)[0]))
    for i in range(EP.shape[0]):
        e = EP[i,:dim_e]
        p = EP[i,dim_e:]
        # print np.argmin(som_e.distances(e)), som_e.distances(e)
        filter_e.learn(e)
        # print "filter_e.winner(e)", filter_e.winner(e)
        # filter_p.learn(p)
        # print "filter_e.activity.shape", filter_e.activity.shape
        # import pdb; pdb.set_trace()
        if use_activity:
            e2p_activation = np.dot(hebblink_filter.T, filter_e.activity.reshape((np.prod(filter_e.map._shape), 1)))
            filter_p.activity = np.clip((e2p_activation / np.sum(e2p_activation)).reshape(filter_p.map._shape), 0, np.inf)
        else:
            e2p_activation = np.dot(hebblink_filter.T, filter_e.distances(e).flatten().reshape(e_shape))
        # print "e2p_activation.shape, np.sum(e2p_activation)", e2p_activation.shape, np.sum(e2p_activation)
        # print "filter_p.activity.shape", filter_p.activity.shape
        # print "np.sum(filter_p.activity)", np.sum(filter_p.activity), (filter_p.activity >= 0).all()

        # filter_p.learn(p)
        emode = 0 # 1, 2
        if i % 1 == 0:
            if emode == 0:
                e2p_w_p_weights_ = []
                for k in range(sampling_search_num):
                    e2p_w_p_weights = filter_p.neuron(filter_p.flat_to_coords(filter_p.sample(1)[0]))
                    e2p_w_p_weights_.append(e2p_w_p_weights)
                pred = np.array(e2p_w_p_weights_)
                # print "pred", pred
                pred_err = np.linalg.norm(pred - p, 2, axis=1)
                # print "np.linalg.norm(e2p_w_p_weights - p, 2)", np.linalg.norm(e2p_w_p_weights - p, 2)
                e2p_w_p = np.argmin(pred_err)
                print "pred_err", e2p_w_p, pred_err[e2p_w_p]
                e2p_w_p_weights = e2p_w_p_weights_[e2p_w_p]
            elif emode == 1:
                if use_activity:
                    e2p_w_p = np.argmax(e2p_activation)
                else:
                    e2p_w_p = np.argmin(e2p_activation)
                e2p_w_p_weights = filter_p.neuron(filter_p.flat_to_coords(e2p_w_p))
                    
            elif emode == 2:
                e2p_w_p = filter_p.winner(p)
                e2p_w_p_weights = filter_p.neuron(filter_p.flat_to_coords(e2p_w_p))
        P_[i] = e2p_w_p_weights
        E_[i] = environment.compute_sensori_effect(P_[i])

        # print "e * hebb", e2p_w_p, e2p_w_p_weights
        
    pl.ioff()

    err_extero = EP[:,:dim_e] - E_
    print "final error MSE", np.mean(np.square(err_extero))
    
    pl.subplot(211)
    pl.title("Extero measured and predicted by E2P with P prediction reexecuted through system")
    pl.plot(EP[:,:dim_e])
    pl.plot(E_)
    pl.subplot(212)
    pl.title("Proprio measured and predicted by E2P")
    pl.plot(EP[:,dim_e:])
    pl.plot(P_)
    
    # pl.subplot(412)
    # pl.title("Proprio dim 1")
    # pl.plot(EP[:,2])
    # pl.plot(P_[:,0])
    # pl.subplot(413)
    # pl.title("Proprio dim 2")
    # pl.plot(EP[:,3])
    # pl.plot(P_[:,1])
    # pl.subplot(414)
    # pl.title("Proprio dim 3")
    # pl.plot(EP[:,4])
    # pl.plot(P_[:,2])
    pl.show()
    
    # # plot neuron heatmaps        
    # ax = pl.subplot(221)
    # pl.title("som_e neuron heatmap")
    # img = som_e.neuron_heatmap()
    # # img= som_e.distance_heatmap(EP[i,:2], axes=(0,1))
    # pl.imshow(img,interpolation="none")
    # ax = pl.subplot(222)
    # pl.title("som_p neuron heatmap")
    # img = som_p.neuron_heatmap()
    # # img= som_e.distance_heatmap(EP[i,:2], axes=(0,1))
    # pl.imshow(img,interpolation="none")

    # pl.subplot(223)
    # pl.title("filter_e neuron heatmap")
    # img = filter_e.map.neuron_heatmap()
    # pl.imshow(img,interpolation="none")
    # pl.subplot(224)
    # pl.title("filter_p neuron heatmap")
    # img = filter_p.map.neuron_heatmap()
    # pl.imshow(img,interpolation="none")
    # pl.show()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", type=str, help="datafile containing t x (dim_extero + dim_proprio) matrix ", default="data/simplearm_n1000/EP_1000.npy")
    args = parser.parse_args()
    
    # main_era(args)
    # main_neupy(args)
    main_kohonen(args)
