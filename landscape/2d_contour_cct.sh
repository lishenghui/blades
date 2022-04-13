
# ------------------------------------------------------------
# SGD
#------------------------------------------------------------
mpirun -n 24 python plot_surface.py --x=-1:3.5:21 --y=-0.2:0.2:21 --model cct_2_3x2_32 \
--model_file cifar10/trained_nets/gm/cct.7t \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  \
 --main_dir_model cifar10/trained_nets/gm/updates/truemean_cct.7t

#  --dir_file cifar10/trained_nets/gm/PCA_weights/directions.h5 \


