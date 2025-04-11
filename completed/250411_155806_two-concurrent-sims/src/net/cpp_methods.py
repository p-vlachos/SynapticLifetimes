
from brian2.units import ms,mV,second
from brian2 import implementation, check_units


# Implementation of synaptic scaling
#
#
@implementation('cpp', headers=['"output_files.h"'], code=r'''
    double syn_scale(double a, double vANormTar, double Asum_post, double veta_scaling, double t, int syn_active, double tRec_start, double tRec_max, int i, int j) {
      
      double a_out;

      if (Asum_post==0.){
          a_out = 0.;
      }
      else{
          a_out = a*(1 + veta_scaling*(vANormTar/Asum_post-1));
      }

      if (t > tRec_start && t < tRec_max && syn_active==1) {
          scaling_deltas_output_file() << t << "," << a << "," << a_out << "," << i << "," << j << "\n";
      }

      return a_out;
    } ''')
@check_units(a=1, vANormTar=1, Asum_post=1, eta_scaling=1, t=second, syn_active=1, tRec_start=second, tRec_max=second, i=1, j=1, result=1)
def syn_scale(a, vANormTar, Asum_post, eta_scaling, t, syn_active, tRec_start, tRec_max, i, j):
    return -1.



# Implementation of E<-I synaptic scaling
#
#
@implementation('cpp', headers=['"output_files.h"'], code=r'''
    double syn_EI_scale(double a, double vANormTar, double Asum_post, double veta_scaling, double t, int syn_active, double tRec_start, double tRec_max, int i, int j) {
      
      double a_out;

      if (Asum_post==0.){
          a_out = 0.;
      }
      else{
          a_out = a*(1 + veta_scaling*(vANormTar/Asum_post-1));
      }

      if (t > tRec_start && t < tRec_max && syn_active==1) {
          scaling_deltas_EI_output_file() << t << "," << a << "," << a_out << "," << i << "," << j << "\n";
      }

      return a_out;
    } ''')
@check_units(a=1, vANormTar=1, Asum_post=1, eta_scaling=1, t=second, syn_active=1, tRec_start=second, tRec_max=second, i=1, j=1, result=1)
def syn_EI_scale(a, vANormTar, Asum_post, eta_scaling, t, syn_active, tRec_start, tRec_max, i, j):
    return -1.

# recording of turnover
#
#
@implementation('cpp',  headers=['"output_files.h"'], code=r'''
    double record_turnover(double t, int was_active_before, int should_become_active, int should_stay_active, int syn_active, int i, int j) {

      if (int(was_active_before==0)*should_become_active==1){
          turnover_output_file() << 1 << "," << t << "," << i << "," << j << "\n";
      }
      else if (was_active_before*int(should_stay_active==0)){
          turnover_output_file() << 0 << "," << t << "," << i << "," << j << "\n";
      }

      return 0.0; // we need to return a dummy value
    } ''')

@check_units(t=second, was_active_before=1, should_become_active=1,
             should_stay_active=1, syn_active=1, i=1, j=1, result=1)
def record_turnover(t, was_active_before, should_become_active,
                    should_stay_active, syn_active, i, j):
    return 0.0






# recording of E<-I turnover
#
#
@implementation('cpp', headers=['"output_files.h"'], code=r'''
    double record_turnover_EI(double t, int was_active_before, int should_become_active, int should_stay_active, int syn_active, int i, int j) {

      if (int(was_active_before==0)*should_become_active==1){
          turnover_EI_output_file() << 1 << "," << t << "," << i << "," << j << "\n";
      }
      else if (was_active_before*int(should_stay_active==0)){
          turnover_EI_output_file() << 0 << "," << t << "," << i << "," << j << "\n";
      }

      return 0.0; // we need to return a dummy value
    } ''')

@check_units(t=second, was_active_before=1, should_become_active=1,
             should_stay_active=1, syn_active=1, i=1, j=1, result=1)
def record_turnover_EI(t, was_active_before, should_become_active,
                    should_stay_active, syn_active, i, j):
    return 0.0



# record spk
#
#
@implementation('cpp', headers=['"output_files.h"'], code=r'''
    double record_spk(double t, int i, int j, double a, double Apre, double Apost, int syn_active, int preorpost, double tRec_start, double tRec_max) {

       if (t > tRec_start && t < tRec_max) {

         if (syn_active > 0){
            spk_register_output_file() << t << "," << i << "," << j << "," << a << "," << Apre << "," << Apost << "," << preorpost << "\n";
         }
   
      }
      return 0.0; // we need to return a dummy value
    } ''')

@check_units(t=second, i=1, j=1, a=1, Apre=1, Apost=1, syn_active=1,
             preorpost=1, tRec_start=second, tRec_max=second, result=1)
def record_spk(t, i, j, a, Apre, Apost, syn_active, preorpost, tRec_start, tRec_max):
    return 0.0


# record spk E<-I
#
#
@implementation('cpp', headers=['"output_files.h"'], code=r'''
    double record_spk_EI(double t, int i, int j, double a, double Apre, double Apost, int syn_active, int preorpost, double tRec_start, double tRec_max) {

       if (t > tRec_start && t < tRec_max) {

         if (syn_active > 0){
            spk_register_EI_output_file() << t << "," << i << "," << j << "," << a << "," << Apre << "," << Apost << "," << preorpost << "\n";
         }
   
      }
      return 0.0; // we need to return a dummy value
    } ''')

@check_units(t=second, i=1, j=1, a=1, Apre=1, Apost=1, syn_active=1,
             preorpost=1, tRec_start=second, tRec_max=second, result=1)
def record_spk_EI(t, i, j, a, Apre, Apost, syn_active, preorpost, tRec_start, tRec_max):
    return 0.0
