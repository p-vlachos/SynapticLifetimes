
from decimal import Decimal
from brian2.units import mV, ms, second, Hz

def netw_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{Network configuration}'

    ax.text(0., 1.0, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)


    text = '$N_e$=' +str(int(nsp['N_e'])) + ', $\, N_i$=%d' %(int(nsp['N_i'])) +\
           '\n $p_{ee}$ = %.2f' % (nsp['p_ee']) +\
           ', $\, p_{ie}$ = %.2f' % (nsp['p_ie']) +\
           '\n $p_{ei}$ = %.2f' % (nsp['p_ei']) +\
           ', $\, p_{ii}$ = %.2f' % (nsp['p_ii'])

    ax.text(0., 0.85, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    
    text = r'\textbf{Simulation}'

    ax.text(0., 0.365, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    
    text = '$T_1 = '+r'\text{\SI{'+'%d}{s}' %(int(nsp['T1']/second))+'}$, '+\
           '$T_2 = '+r'\text{\SI{'+'%d}{s}' %(int(nsp['T2']/second))+'}$, '+\
           '$T_3 = '+r'\text{\SI{'+'%d}{s}' %(int(nsp['T3']/second))+'}$' +\
           '\n$T_4 = '+r'\text{\SI{'+'%d}{s}' %(int(nsp['T4']/second))+'}$, '+\
           '$T_5 = '+r'\text{\SI{'+'%d}{s}' %(int(nsp['T5']/second))+'}$, ' +\
           '$dt='+r'\text{'+'\SI{%.2f}{ms}' %(nsp['dt']/ms) +'}$'


    ax.text(0., 0.215, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    


def neuron_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{Neuron parameters}'

    ax.text(0., 1.0, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    
    text = '$'+r'\tau'+'$=\SI{%.0f}{ms}' %(nsp['tau']/ms) +\
           ', $\,E_l$= \SI{%.0f}{mV}' % (nsp['El']/mV) +\
           '\n $\, E_e$= \SI{%.0f}{mV}' % (nsp['Ee']/mV) +\
           '$,\, E_i$= \SI{%.0f}{mV}' % (nsp['Ei']/mV) +\
           '\n $'+r'\,\,\,\,\,\, \tau_e'+'$=\SI{%.0f}{ms}' % (nsp['tau_e']/ms) +\
           ', $'+r'\tau_i'+'$=\SI{%.0f}{ms}' % (nsp['tau_i']/ms)

    ax.text(0., 0.85, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)
        

       
  
    text = r'\textbf{excitatory, inhibitory}'

    ax.text(0., 0.365, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    
    text = '$V^T_{e}$=\SI{%.2f}{mV}' % (nsp['Vt_e']/mV) +\
           ', $V^T_{i}$=\SI{%.2f}{mV}' % (nsp['Vt_i']/mV) +\
           '\n $V^r_{e}$=\SI{%.0f}{mV}' % (nsp['Vr_e']/mV) +\
           ', $V^r_{i}$=\SI{%.0f}{mV}' % (nsp['Vr_i']/mV) 


    ax.text(0., 0.2, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)



    


def synapse_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{Synapse parameters}'

    ax.text(0., 0.9, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)


    
    text = '$a_{ee}=$ %.2E ' %(Decimal(nsp['a_ee']/nsp['ascale'])) +\
           '$a_{\mathrm{scale}}$' +\
           '\n$a_{ie}=$ %.2E ' %(Decimal(nsp['a_ie']/nsp['ascale'])) +\
           '$a_{\mathrm{scale}}$' +\
           '\n $a_{ei}$ = %.2E ' % (Decimal(nsp['a_ei']/nsp['ascale'])) +\
           '$a_{\mathrm{scale}}$' +\
           '\n $a_{ii}$ = %.2E '  % Decimal((nsp['a_ii']/nsp['ascale']))  +\
           '$a_{\mathrm{scale}}$' +\
           '\n$a_{\mathrm{scale}} =$ %.3E' % Decimal(nsp['ascale'])
                                            
    ax.text(0., 0.75, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)
        

       
  

def stdp_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{STDP parameters}'

    ax.text(0., 1.0, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)
    
    
    text = r'$\tau'+'_{\mathrm{pre}} =\,$ \SI{%.2f}{ms}' %(nsp['taupre']/ms) +\
            r',$\,\tau'+'_{\mathrm{post}}=\,$ \SI{%.2f}{ms}' %(nsp['taupost']/ms) +\
            '\n $A_{\mathrm{plus}}\,\,\,$= $\,\,\,\,%f$' %(nsp['Aplus']) + \
            '\n $A_{\mathrm{minus}}$= $%f$' %(nsp['Aminus']) +\
            '\n $a_{\mathrm{max}} = '+'%f' %(nsp['amax']) +'$'

    ax.text(0., 0.85, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    if nsp['istdp_active']:

        text = r'\textbf{iSTDP parameters}'
        
        ax.text(0., 0.2, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=13,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)

        text = r'$\mathrm{LTD}_{\alpha} = %f $' %(nsp['LTD_a']) 

        ax.text(0., 0.05, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=12,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)

        

   

    
def sn_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{Synaptic scaling}'

    ax.text(0., 1.0, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    text = r'$a^{ee}_{\mathrm{TotalMax}} = %f $' %(nsp['ATotalMax']) +\
           '\n $\Delta_{\mathrm{scaling}} =\,$' +\
           '\SI{%.2f}{ms}' %(nsp['dt_synEE_scaling']/ms) + \
           '\n $\eta_{\mathrm{scaling}} = %f$' %(nsp['eta_scaling'])

    ax.text(0., 0.85, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)

    if nsp['iscl_active']:

        text = r'\textbf{Inh.~synaptic scaling}'
        
        ax.text(0., 0.30, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=13,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)

        text = r'$a^{ei}_{\mathrm{TotalMax}} = %f $' %(nsp['iATotalMax']) 

        ax.text(0., 0.15, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=12,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)


    

def strct_params_display(ax, bpath, nsp):

    ax.axis('off')

    text = r'\textbf{Structural plasticity parameters}'

    ax.text(0., 1.0, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=13,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)


    text = '$p_{\mathrm{insert}} = %f$' %(nsp['insert_P']) +\
           '\n$a_{\mathrm{insert}} = %f$' %(nsp['a_insert']) +\
           '\n$c = %f$' %(nsp['strct_c']) +\
           '\n$p_{\mathrm{inactivate}} = %f$' %(nsp['p_inactivate']) +\
           '\n $a_{\mathrm{thrshld}} = %f$' %(nsp['prn_thrshld']) +\
           '\n $\Delta_{\mathrm{strct}} =\,$ \SI{%.2f}{ms}' %(nsp['strct_dt']/ms)

    if nsp['istrct_active']==1:
        
        text = '$p^{\mathrm{EE}}_{\mathrm{insert}} = %f$' %(nsp['insert_P']) +\
               ', $p^{\mathrm{EI}}_{\mathrm{insert}} = %f$' %(nsp['insert_P_ei']) +\
               '\n$a^{\mathrm{EE}}_{\mathrm{insert}} = %f$' %(nsp['a_insert']) +\
               ', $a^{\mathrm{EI}}_{\mathrm{insert}} = %f$' %(nsp['a_insert']) +\
               '\n$c^{EE} = %f$' %(nsp['strct_c']) +\
               ', $c^{EI} = %f$' %(nsp['strct_c']) +\
               '\n$p^{\mathrm{EE}}_{\mathrm{inactivate}} = %f$' %(nsp['p_inactivate']) +\
               ', $p^{\mathrm{EI}}_{\mathrm{inactivate}} = %f$' %(nsp['p_inactivate']) +\
               '\n $a_{\mathrm{thrshld}} = %f$' %(nsp['prn_thrshld']) +\
               '\n $\Delta_{\mathrm{strct}} =\,$ \SI{%.2f}{ms}' %(nsp['strct_dt']/ms)
        

    ax.text(0., 0.85, text,
            horizontalalignment='left',
            verticalalignment='top',
            linespacing = 1.95,
            fontsize=12,
            bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                  'alpha':1, 'edgecolor':'none'},
            transform = ax.transAxes)
        

def poisson_input_params_display(ax, bpath, nsp):    
    
    ax.axis('off')

    if nsp['external_mode']=='poisson':

        text = r'\textbf{Poisson input parameters}'

        ax.text(0., 1.0, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=13,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)


        text = '$N_{\mathrm{poisson}} = %d$' %(int(nsp['NPInp'])) +\
               '\n'+r'$\mathrm{rate} = \text{\SI{'+'%.3f}{Hz}}$' \
                  %(nsp['PInp_rate']/Hz) +\
               '\n' +'$N_{\mathrm{poisson}} = %d$' %(int(nsp['NPInp_inh'])) +\
               '\n'+r'$\mathrm{rate} = \text{\SI{'+'%.3f}{Hz}}$' \
                  %(nsp['PInp_inh_rate']/Hz) 

               # '\n$a_{\mathrm{epoi} = %f$' %(nsp['a_EPoi']) +\
               # '\n$a_{\mathrm{ipoi} = %f$' %(nsp['a_IPoi']) +\
               # '\n$p_{\mathrm{epoi} = %f$' %(nsp['p_EPoi']) +\
               # '\n$p_{\mathrm{ipoi} = %f$' %(nsp['p_IPoi'])


        ax.text(0., 0.85, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=12,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)

    elif nsp['external_mode']=='memnoise':


        text = r'\textbf{Membrane noise parameters}'

        ax.text(0., 1.0, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=13,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)


        text = '$\mu_{\mathrm{e}} =' + r'\text{\SI{' + \
               '%.2f}{mV}}$, ' %(nsp['mu_e']/mV) + \
               '$\mu_{\mathrm{i}} =' + r'\text{\SI{' + \
               '%.2f}{mV}}$, ' %(nsp['mu_i']/mV) + \
               '\n' + '$\sigma_{\mathrm{e}} =' + r'\text{\SI{' + \
               '%.2f}{mV}}$, ' %(nsp['sigma_e']/mV) + \
               '$\sigma_{\mathrm{i}} =' + r'\text{\SI{' + \
               '%.2f}{mV}}$, ' %(nsp['sigma_i']/mV) 
               
        ax.text(0., 0.85, text,
                horizontalalignment='left',
                verticalalignment='top',
                linespacing = 1.95,
                fontsize=12,
                bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
                      'alpha':1, 'edgecolor':'none'},
                transform = ax.transAxes)
