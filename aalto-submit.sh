#! /bin/tcsh

#set ff=( c_in12_rn152_pool5o_d_a i3d-25-128-avg \
#         c_in12_rn152_pool5o_d_a,i3d-25-128-avg )

set ff=( i3d-25-128-avg,audioset-527 )

set hh=`seq 20 20 500`

foreach h ( $hh )
    foreach f ( $ff )
	sbatch aalto-sbatch.sh $h $f
    end
end


