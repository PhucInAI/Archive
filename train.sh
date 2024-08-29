python train_la.py --config configs/LA_ESFPNet_B0_Kvasir.yaml -d 0 -n LA_ESFPNet_B0_Kvasir
0.906192
python train_la.py --config configs/LA_ESFPNet_B0_CVC-ClinicDB.yaml -d 1 -n LA_ESFPNet_B0_CVC-ClinicDB
0.919352


python train_la_ffesnet.py --config configs/LA_FFESNet_B0_Kvasir.yaml -d 0 -n LA_FFESNet_B0_Kvasir
1.0 0.908344
python train_la_ffesnet.py --config configs/LA_FFESNet_B0_CVC-ClinicDB.yaml -d 1 -n LA_FFESNet_B0_CVC-ClinicDB
0.5 0.918575
1.0 0.932015
0.929590

python train_la_ffesnet.py --config configs/LA_FFESNet_B4_Kvasir.yaml -d 0 -n LA_FFESNet_B4_Kvasir

python train_la_ffesnet.py --config configs/LA_FFESNet_B4_CVC-ClinicDB.yaml -d 1 -n LA_FFESNet_B4_CVC-ClinicDB

python train_GA_n_PB_ffesnet.py --config configs/FFESNet_B4.yaml -d 0 > FFESNet_B4_Kvasir.log

python train_GA_n_PB_ffesnet.py --config configs/FFESNet_B4_CSP.yaml -d 1