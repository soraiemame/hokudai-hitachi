#!/usr/bin/env bash
echo "Building 'judge/judge'..."
g++ -std=c++17 -O2 -I. -Ilib -Ijudge -Wno-format-truncation judge/all_in_one.cpp -o judge/judge
# g++ -std=c++17 -O2 -I. -Ilib -lstdc++fs -Ijudge -Wno-format-truncation judge/all_in_one.cpp -o judge/judge
echo "Done."

echo "Building 'generator/trans_prob_mat_generator'..."
g++ -std=c++17 -O2 -I. -Ilib -Ijudge generator/generate_trans_mat.cpp -o generator/trans_prob_mat_generator
echo "Done."

echo "Building 'generator/map_generator'..."
g++ -std=c++17 -O2 -I. -Ilib -Ijudge generator/random_map.cpp -o generator/map_generator
echo "Done."

echo "Building 'generator/calc_dist'..."
g++ -std=c++17 -O2 -I. -Ilib -Ijudge generator/calc_dist.cpp -o generator/calc_dist
echo "Done."

echo "Building 'sample/sample_A.cpp'..."
g++ -std=c++17 -O2 sample/sample_A.cpp -o sample/sample_A
echo "Done."

echo "Building 'sample/sample_B.cpp'..."
g++ -std=c++17 -O2 sample/sample_B.cpp -o sample/sample_B
echo "Done."
