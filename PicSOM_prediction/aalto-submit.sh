#! /bin/tcsh

#set ff=( c_in12_rn152_pool5o_d_a i3d-25-128-avg \
#         c_in12_rn152_pool5o_d_a,i3d-25-128-avg )

#set ff=( i3d-25-128-avg,audioset-527 )
#set ff=( c3d-rn18-s1m-pool5-a )
#set ff=( i3d-25-128-avg,audioset-527,c3d-rn18-s1m-pool5-a )
#set ff=( i3d-25-128-avg,audioset-527,c_in12_rn152_pool5o_d_a )
#set ff=( C3D )
#set ff=( C3D,audioset-527 )
#set ff=( i3d-25-128-avg )
#set ff=( i3d-25-128-avg,audioset-527 )
#set ff=( C3D audioset-527 C3D,audioset-527 i3d-25-128-avg i3d-25-128-avg,audioset-527 )
#set ff=( bert3 C3D,bert3 C3D,audioset-527,bert3 i3d-25-128-avg,bert3 i3d-25-128-avg,audioset-527,bert3 )
set ff=( i3d-25-128-pred i3d-25-128-pred,audioset-527 i3d-25-128-pred,bert3 i3d-25-128-pred,i3d-25-128-avg i3d-25-128-pred,i3d-25-128-avg,bert3 i3d-25-128-pred,i3d-25-128-avg,audioset-527,bert3 )

set hh=`seq 20 20 800`

foreach h ( $hh )
    foreach f ( $ff )
	sbatch aalto-sbatch.sh trecvid/train/short // $h $f 1000
    end
end


