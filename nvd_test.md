TODO: 整个流程的指令(文档+sh脚本)
TODO：--prefix
TODO: 解释ComplexGen Trim的代码未开源，找作者要
TODO: evaluate运行的指令

# Evaluation 

## Generate the result of geomtry refinement

1. Please sample **uniformly** on the mesh for 10000 point_clouds
2. Move your point_clouds to `data/defalut/test_point_clouds`, the naming format is: `id_10000.ply`
3. Write the model ids you want to generate in the file named `test_ids.txt`, with each model's id on a new line
4. Run `python test.py`
5. You can get `id_geom_refine.json` and `id_geom_refine.obj` for evaluation in `experiments/default/test_obj` folder

## sample on the `geom_refine.json`

Run `python sample_on_geom_refine.py` to get the data for evaluation

After that, you will find the following directory structure for **PATH_TO_OUTPUT**:

- **cut_grouped(mesh, only generate while trim version)**
  - *\_cut_grouped.ply: Output of complexgen trim (Rescaled by **sample_geom_refine_and_scale.py**)

- **geom_refine(mesh)**
  - *\_geom_refine.ply: Output of complexgen geometry refinement (Rescaled by **sample_geom_refine_and_scale.py**)

- **sample_on_cut_grouped(points, only generate while trim version)**
  - \*.ply: Sampled points based on area on the output of complexgen trim (Generated by **prepare_complex_trim_data**)

- **sample_on_geom_refine(points)**
  - **vertices**
    - \*.ply: Vertices prediction from complexgen geometry refinement json (Generated by **sample_geom_refine_and_scale.py**)
  - **curves**
    - \*.ply: Sampled points based on curve length on the output of complexgen geometry refinement json (Generated by **sample_geom_refine_and_scale.py**)
  - **surfaces**
    - \*.ply: Sampled points based on area on the output of complexgen geometry refinement json (Generated by **sample_geom_refine_and_scale.py**)

- **topo**
  - **.txt**: Topology of ComplexGen generated by `_geom_refine.json`

# Evaluate one model

Similar to the above, just modify `test_ids.txt`, which contains only one model id. Then follow the instructions above.

# ComplexGen Trim Evaluation

Since the author did not open source the Trim code, the non-trim version is used for evaluation. If you want to get the trim code, please contact the author yourself.

After you get the code, you should replace the suffix `_geom_refine.json` with `_extraction_final.json`, `_geom_refine.obj` with `_extraction_final.obj`