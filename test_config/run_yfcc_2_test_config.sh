JSONS=( 
yfcc_2_80nodes_dump_feat_0.json
yfcc_2_80nodes_dump_feat_1.json 
yfcc_2_80nodes_dump_feat_2.json
yfcc_2_80nodes_dump_feat_3.json
yfcc_2_80nodes_dump_feat_4.json
yfcc_2_80nodes_dump_feat_5.json
yfcc_2_80nodes_dump_feat_6.json
yfcc_2_80nodes_dump_feat_7.json
yfcc_2_80nodes_dump_feat_8.json
yfcc_2_80nodes_dump_feat_9.json
yfcc_2_80nodes_dump_feat_10.json
yfcc_2_80nodes_dump_feat_11.json
yfcc_2_80nodes_dump_feat_12.json
yfcc_2_80nodes_dump_feat_13.json
yfcc_2_80nodes_dump_feat_14.json
yfcc_2_80nodes_dump_feat_15.json
yfcc_2_80nodes_dump_feat_16.json )

for JSON in $JSONS
do
    python ./data/cache_yfcc_test_node_edge_feat.py --config ./test_config/$JSON --dev 0
done