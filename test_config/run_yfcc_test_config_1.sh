JSONS=( yfcc_80nodes_dump_feat_1.json yfcc_80nodes_dump_feat_3.json yfcc_80nodes_dump_feat_5.json yfcc_80nodes_dump_feat_7.json yfcc_80nodes_dump_feat_9.json yfcc_80nodes_dump_feat_11.json yfcc_80nodes_dump_feat_13.json yfcc_80nodes_dump_feat_15.json )

for JSON in $JSONS
do
    python ./data/cache_onedsfm_test_node_edge_feat.py --config ./test_config/$JSON --dev 1
done