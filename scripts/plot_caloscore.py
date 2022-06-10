import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from CaloScore import CaloScore


hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


rank = hvd.rank()
size = hvd.size()

utils.SetStyle()


parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
parser.add_argument('--nevts', type=float,default=1e5, help='Number of events to load')
parser.add_argument('--nslices', type=int,default=16, help='Number of files generated')
parser.add_argument('--nrank', type=int,default=0, help='Rank of the files generated')
parser.add_argument('--batch_size', type=int,default=50, help='Batch size for generation')
parser.add_argument('--model', default='VPSDE', help='Diffusion model to load. Options are: VPSDE, VESDE,  subVPSDE, all')
parser.add_argument('--sample', action='store_true', default=False,help='Sample from learned model')
parser.add_argument('--comp_eps', action='store_true', default=False,help='Load files with different eps')
parser.add_argument('--comp_N', action='store_true', default=False,help='Load files with different N')

flags = parser.parse_args()

nevts = int(flags.nevts)
dataset_config = utils.LoadJson(flags.config)
emax = dataset_config['EMAX']
emin = dataset_config['EMIN']
run_classifier=False

if flags.sample:
    checkpoint_folder = '../checkpoints_{}_{}'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    effective_rank=flags.nrank*size+rank
    energies = []
    for dataset in dataset_config['EVAL']:
        e_ = utils.EnergyLoader(os.path.join(flags.data_folder,dataset),
                                flags.nevts,
                                rank=effective_rank,
                                emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
                                logE=dataset_config['logE'])
        energies.append(e_)

    energies = np.reshape(energies,(-1,1))
    #print(energies)
    model = CaloScore(dataset_config['SHAPE_PAD'][1:],energies.shape[1],nevts,sde_type=flags.model,config=dataset_config)    
    model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
    batch_size = flags.batch_size


    generated=model.PCSampler(cond=energies,
                              snr=dataset_config['SNR'],
                              num_steps=dataset_config['NSTEPS']).numpy()
    generated,energies = utils.ReverseNorm(generated,energies[:nevts],
                                           shape=dataset_config['SHAPE'],
                                           logE=dataset_config['logE'],
                                           max_deposit=dataset_config['MAXDEP'],
                                           norm_data=dataset_config['NORMED'],
                                           emax = dataset_config['EMAX'],emin = dataset_config['EMIN'])
    generated[generated<dataset_config['ECUT']] = 0 #min from samples

    with h5.File(os.path.join(flags.data_folder,dataset_config['EVAL'][0].replace('.hdf5','_mask.hdf5')),"r") as h5f:
        #mask file for voxels that are always empty, run utils.py to create a new one
        mask = h5f['mask'][:]
    generated = generated*(np.reshape(mask,(1,-1))==0)
    
    with h5.File(os.path.join(flags.data_folder,'generated_{}_{}_{}.h5'.format(dataset_config['CHECKPOINT_NAME'],flags.model,effective_rank)),"w") as h5f:
        dset = h5f.create_dataset("showers", data=1000*np.reshape(generated,(generated.shape[0],-1)))
        dset = h5f.create_dataset("incident_energies", data=1000*energies)
else:
    def LoadSamples(model,nrank=16):
        generated = []
        energies = []
        for rank in range(nrank):        
            with h5.File(os.path.join(flags.data_folder,'generated_{}_{}_{}.h5'.format(dataset_config['CHECKPOINT_NAME'],model,rank)),"r") as h5f:
                generated.append(h5f['showers'][:]/1000.)
                energies.append(h5f['incident_energies'][:]/1000.)
        energies = np.reshape(energies,(-1,1))
        generated = np.reshape(generated,dataset_config['SHAPE'])
        return generated,energies


    if flags.model != 'all':
        models = [flags.model]
        if flags.comp_eps:
            variations = ['0p0','0p3']
            models += ["{}_{}".format(variation,flags.model) for variation in variations]
        elif flags.comp_N:
            variations = ['50','500']
            models += ["{}_{}".format(variation,flags.model) for variation in variations]
    else:
        models = ['VPSDE','subVPSDE','VESDE']

    energies = []
    data_dict = {}
    for model in models:
        if np.size(energies) == 0:
            data,energies = LoadSamples(model,flags.nslices)
            data_dict[utils.name_translate[model]]=data
        else:
            data_dict[utils.name_translate[model]]=LoadSamples(model,flags.nslices)[0]
    total_evts = energies.shape[0]

    
    data = []
    true_energies = []
    for dataset in dataset_config['EVAL']:
        with h5.File(os.path.join(flags.data_folder,dataset),"r") as h5f:
            data.append(h5f['showers'][:total_evts]/1000.)
            true_energies.append(h5f['incident_energies'][:total_evts]/1000.)

    
    data_dict['Geant4']=np.reshape(data,dataset_config['SHAPE'])
    true_energies = np.reshape(true_energies,(-1,1))


    
    #Plot high level distributions and compare with real values
    assert np.allclose(true_energies,energies), 'ERROR: Energies between samples dont match'


    def ScatterESplit(data_dict,true_energies):
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        fig,ax = SetFig("Gen. energy [GeV]","Dep. energy [GeV]")
        for key in data_dict:
            ax.scatter(true_energies[10000:10500],
                       np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)[10000:10500],
                       label=key)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc='best',fontsize=16,ncol=1)
        fig.savefig('{}/FCC_Scatter_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))


    def AverageShowerWidth(data_dict):
        eta_bins = dataset_config['SHAPE'][2]
        eta_binning = np.linspace(-1,1,eta_bins+1)
        eta_coord = [(eta_binning[i] + eta_binning[i+1])/2.0 for i in range(len(eta_binning)-1)]

        def GetMatrix(sizex,sizey,minval=-1,maxval=1):
            nbins = sizex
            binning = np.linspace(minval,maxval,nbins+1)
            coord = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
            matrix = np.repeat(np.expand_dims(coord,-1),sizey,-1)
            return matrix

        
        eta_matrix = GetMatrix(dataset_config['SHAPE'][2],dataset_config['SHAPE'][3])
        eta_matrix = np.reshape(eta_matrix,(1,1,eta_matrix.shape[0],eta_matrix.shape[1],1))
        
        
        phi_matrix = np.transpose(GetMatrix(dataset_config['SHAPE'][3],dataset_config['SHAPE'][2]))
        phi_matrix = np.reshape(phi_matrix,(1,1,phi_matrix.shape[0],phi_matrix.shape[1],1))

        def GetCenter(matrix,energies,power=1):
            ec = energies*np.power(matrix,power)
            sum_energies = np.sum(np.reshape(energies,(energies.shape[0],energies.shape[1],-1)),-1)
            ec = np.reshape(ec,(ec.shape[0],ec.shape[1],-1)) #get value per layer
            ec = np.ma.divide(np.sum(ec,-1),sum_energies).filled(0)

            return ec

        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width

        
        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_eta = {}
        feed_dict_eta2 = {}
        
        for key in data_dict:
            feed_dict_phi[key] = GetCenter(phi_matrix,data_dict[key])
            feed_dict_phi2[key] = GetWidth(feed_dict_phi[key],GetCenter(phi_matrix,data_dict[key],2))
            feed_dict_eta[key] = GetCenter(eta_matrix,data_dict[key])
            feed_dict_eta2[key] = GetWidth(feed_dict_eta[key],GetCenter(eta_matrix,data_dict[key],2))
            

        fig,ax0 = utils.PlotRoutine(feed_dict_eta,xlabel='Layer number', ylabel= 'x-center of energy')
        fig.savefig('{}/FCC_EtaEC_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi,xlabel='Layer number', ylabel= 'y-center of energy')
        fig.savefig('{}/FCC_PhiEC_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_eta2,xlabel='Layer number', ylabel= 'x-width')
        fig.savefig('{}/FCC_EtaW_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi2,xlabel='Layer number', ylabel= 'y-width')
        fig.savefig('{}/FCC_PhiW_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))

        return feed_dict_eta2

    def AverageELayer(data_dict):
        
        def _preprocess(data):
            preprocessed = np.reshape(data,(total_evts,dataset_config['SHAPE'][1],-1))
            preprocessed = np.sum(preprocessed,-1)
            #preprocessed = np.mean(preprocessed,0)
            return preprocessed
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean deposited energy [GeV]')
        fig.savefig('{}/FCC_EnergyZ_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def AverageEX(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],dataset_config['SHAPE'][3],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed
            
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='x-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyX_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict
        
    def AverageEY(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,2,1,3,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],dataset_config['SHAPE'][2],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='y-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyY_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def HistEtot(data_dict):
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed,-1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

            
        binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),10)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
        ax0.set_xscale("log")
        fig.savefig('{}/FCC_TotalE_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict
        
    def HistNhits(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed>0,-1)
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits', ylabel= 'Normalized entries',label_loc='upper left')
        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        fig.savefig('{}/FCC_Nhits_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict
    def HistMaxELayer(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],dataset_config['SHAPE'][1],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Max. voxel/Dep. energy')
        fig.savefig('{}/FCC_MaxEnergyZ_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def HistMaxE(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0,1,10)
        fig,ax0 = utils.HistRoutine(feed_dict,ylabel='Normalized entries', xlabel= 'Max. voxel/Dep. energy',binning=binning,logy=True)
        fig.savefig('{}/FCC_MaxEnergy_{}_{}.pdf'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def Plot_Shower_2D(data_dict):
        #cmap = plt.get_cmap('PiYG')
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad("white")
        plt.rcParams['pcolor.shading'] ='nearest'
        layer_number = [10,44]
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        for layer in layer_number:
            
            def _preprocess(data):
                preprocessed = data[:,layer,:]
                preprocessed = np.mean(preprocessed,0)
                preprocessed[preprocessed==0]=np.nan
                return preprocessed

            vmin=vmax=0
            for ik,key in enumerate(['Geant4',utils.name_translate[flags.model]]):
                fig,ax = SetFig("x-bin","y-bin")
                average = _preprocess(data_dict[key])
                if vmax==0:
                    vmax = np.nanmax(average[:,:,0])
                    vmin = np.nanmin(average[:,:,0])
                    print(vmin,vmax)
                im = ax.pcolormesh(range(average.shape[0]), range(average.shape[1]), average[:,:,0], cmap=cmap,vmin=vmin,vmax=vmax)

                yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
                yScalarFormatter.set_powerlimits((0,0))
                #cbar.ax.set_major_formatter(yScalarFormatter)

                cbar=fig.colorbar(im, ax=ax,label='Dep. energy [GeV]',format=yScalarFormatter)
                
                
                bar = ax.set_title("{}, layer number {}".format(key,layer),fontsize=15)

                fig.savefig('{}/FCC_{}2D_{}_{}_{}.pdf'.format(flags.plot_folder,key,layer,dataset_config['CHECKPOINT_NAME'],flags.model))
            

    high_level = []
    plot_routines = {
        # 'Energy per layer':AverageELayer,
        # 'Energy':HistEtot,
        # '2D Energy scatter split':ScatterESplit,
        # 'Nhits':HistNhits,
    }
    
    if '1' in flags.config:
        plot_routines['Max voxel']=HistMaxE
    else:
        pass
        plot_routines['Shower width']=AverageShowerWidth        
        plot_routines['Energy per eta']=AverageEX
        plot_routines['Energy per phi']=AverageEY
        plot_routines['2D average shower']=Plot_Shower_2D
        plot_routines['Max voxel']=HistMaxELayer

        
    for plot in plot_routines:
        if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
        print(plot)
        if 'split' in plot:
            plot_routines[plot](data_dict,energies)
        else:
            high_level.append(plot_routines[plot](data_dict))
            
