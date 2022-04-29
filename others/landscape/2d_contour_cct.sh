
# ------------------------------------------------------------
# SGD
#------------------------------------------------------------
# mpirun -n 24 python plot_surface.py --x=-1:1:5 --y=-1:1:5 --model cct_2_3x2_32 \
# --model_file cifar10/trained_nets/saved_final_model.pt \
# --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  
#  --main_dir_model cifar10/trained_nets/gm/updates/truemean_cct.7t
# --model_file2 cifar10/trained_nets/saved_init_model.pt \
#  --dir_file cifar10/trained_nets/gm/PCA_weights/directions.h5 \


mpirun -n 24 python plot_surface.py --mpi --cuda --model cct_2_3x2_32 --x=-1:1:10 --y=-1:1:10 \
--model_file cifar10/trained_nets/saved_final_model.pt \
--model_file2 cifar10/trained_nets/saved_init_model.pt \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot