JSONS=( yfcc_80nodes_dump_feat_0.json yfcc_80nodes_dump_feat_2.json yfcc_80nodes_dump_feat_4.json yfcc_80nodes_dump_feat_6.json yfcc_80nodes_dump_feat_8.json yfcc_80nodes_dump_feat_10.json yfcc_80nodes_dump_feat_12.json yfcc_80nodes_dump_feat_14.json )

for JSON in $JSONS
do
    python ./data/cache_onedsfm_test_node_edge_feat.py --config ./test_config/$JSON --dev 3
done