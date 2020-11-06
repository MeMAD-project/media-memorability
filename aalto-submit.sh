#! /bin/tcsh

#set ff=( c_in12_rn152_pool5o_d_a i3d-25-128-avg \
#         c_in12_rn152_pool5o_d_a,i3d-25-128-avg )

#set ff=( i3d-25-128-avg,audioset-527 )
#set ff=( c3d-rn18-s1m-pool5-a )
#set ff=( i3d-25-128-avg,audioset-527,c3d-rn18-s1m-pool5-a )
set ff=( i3d-25-128-avg,audioset-527,c_in12_rn152_pool5o_d_a )

set hh=`seq 20 20 500`

foreach h ( $hh )
    foreach f ( $ff )
	sbatch aalto-sbatch.sh short $h $f
    end
end


