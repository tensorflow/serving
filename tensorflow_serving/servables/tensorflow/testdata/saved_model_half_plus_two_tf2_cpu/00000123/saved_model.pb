ݨ:�.
�
a
b
c
	asset
classify_x2y3
classify_xy
predict
regress_x2y3

regress_xy	
regress_xy2


signatures"
_generic_user_object
:	 2a
:	 2b
:	 2c
* 
�
trace_02�
__inference_classify_x2y3_215�*�
�
inputs
���	�
jx2
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
__inference_classify_xy_195�*!�
�
inputs���������
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
__inference_predict_235�*�
�
����
jx
args
 
varargs
 
varkw"�
�*    
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
__inference_regress_x2y3_165�*�
�
inputs
���	�
jx2
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
__inference_regress_xy_115�*!�
�
inputs���������
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
__inference_regress_xy2_145�*!�
�
inputs���������
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
regress_x_to_y
regress_x_to_y2
regress_x2_to_y3
classify_x_to_y
classify_x2_to_y3
serving_default"
signature_map
�B�
__inference_classify_x2y3_215inputs"�*
 
���	�
jx2
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
__inference_classify_xy_195inputs"�*
 
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
__inference_predict_235x"�*
 
����
jx
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
__inference_regress_x2y3_165inputs"�*
 
���	�
jx2
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
__inference_regress_xy_115inputs"�*
 
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
__inference_regress_xy2_145inputs"�*
 
����
jserialized_proto
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
,__inference_signature_wrapper_regress_xy_125inputs"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�

jinputs

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
-__inference_signature_wrapper_regress_xy2_155inputs"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�

jinputs

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
.__inference_signature_wrapper_regress_x2y3_175inputs"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�

jinputs

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
-__inference_signature_wrapper_classify_xy_205inputs"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�

jinputs

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
/__inference_signature_wrapper_classify_x2y3_225inputs"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�

jinputs

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
)__inference_signature_wrapper_predict_245x"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�
jx

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecm
__inference_classify_x2y3_215L"�
�
�
inputs
� ""�

scores�
scores�
__inference_classify_xy_195b+�(
!�
�
inputs���������
� "/�,
*
scores �
scores���������X
__inference_predict_235=�
�
�
x
� "�

y�
yn
__inference_regress_x2y3_165N"�
�
�
inputs
� "$�!

outputs�
outputs�
__inference_regress_xy2_145d+�(
!�
�
inputs���������
� "1�.
,
outputs!�
outputs����������
__inference_regress_xy_115d+�(
!�
�
inputs���������
� "1�.
,
outputs!�
outputs����������
/__inference_signature_wrapper_classify_x2y3_225V,�)
� 
"�

inputs�
inputs""�

scores�
scores�
-__inference_signature_wrapper_classify_xy_205l5�2
� 
+�(
&
inputs�
inputs���������"/�,
*
scores �
scores���������o
)__inference_signature_wrapper_predict_245B"�
� 
�

x�
x"�

y�
y�
.__inference_signature_wrapper_regress_x2y3_175X,�)
� 
"�

inputs�
inputs"$�!

outputs�
outputs�
-__inference_signature_wrapper_regress_xy2_155n5�2
� 
+�(
&
inputs�
inputs���������"1�.
,
outputs!�
outputs����������
,__inference_signature_wrapper_regress_xy_125n5�2
� 
+�(
&
inputs�
inputs���������"1�.
,
outputs!�
outputs���������2%foo.txt

asset_path_initializer:0*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
classify_x2_to_y3{
.
inputs$
classify_x2_to_y3_inputs:0-
scores#
StatefulPartitionedCall:0tensorflow/serving/predict*�
classify_x_to_y�
5
inputs+
classify_x_to_y_inputs:0���������<
scores2
StatefulPartitionedCall_1:0���������tensorflow/serving/predict*�
regress_x2_to_y3}
-
inputs#
regress_x2_to_y3_inputs:00
outputs%
StatefulPartitionedCall_2:0tensorflow/serving/predict*�
regress_x_to_y�
4
inputs*
regress_x_to_y_inputs:0���������=
outputs2
StatefulPartitionedCall_3:0���������tensorflow/serving/predict*�
regress_x_to_y2�
5
inputs+
regress_x_to_y2_inputs:0���������=
outputs2
StatefulPartitionedCall_4:0���������tensorflow/serving/predict*�
serving_default�
"
x
serving_default_x:0*
y%
StatefulPartitionedCall_5:0tensorflow/serving/predict"
x*    "
saved_model_main_op

NoOpL
saver_filename:0StatefulPartitionedCall_6:0StatefulPartitionedCall_78��
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
V
cVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namec
O
c/Read/ReadVariableOpReadVariableOpc*
_output_shapes
: *
dtype0
V
bVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameb
O
b/Read/ReadVariableOpReadVariableOpb*
_output_shapes
: *
dtype0
V
aVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namea
O
a/Read/ReadVariableOpReadVariableOpa*
_output_shapes
: *
dtype0
a
classify_x2_to_y3_inputsPlaceholder*
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallclassify_x2_to_y3_inputsac*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *8
f3R1
/__inference_signature_wrapper_classify_x2y3_225
q
classify_x_to_y_inputsPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCallclassify_x_to_y_inputsab*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *6
f1R/
-__inference_signature_wrapper_classify_xy_205
`
regress_x2_to_y3_inputsPlaceholder*
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCall_2StatefulPartitionedCallregress_x2_to_y3_inputsac*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *7
f2R0
.__inference_signature_wrapper_regress_x2y3_175
p
regress_x_to_y_inputsPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_3StatefulPartitionedCallregress_x_to_y_inputsab*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *5
f0R.
,__inference_signature_wrapper_regress_xy_125
q
regress_x_to_y2_inputsPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_4StatefulPartitionedCallregress_x_to_y2_inputsac*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *6
f1R/
-__inference_signature_wrapper_regress_xy2_155
d
serving_default_x/inputConst*
_output_shapes
:*
dtype0*
valueB*    
~
serving_default_xPlaceholderWithDefaultserving_default_x/input*
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCall_5StatefulPartitionedCallserving_default_xab*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *2
f-R+
)__inference_signature_wrapper_predict_245

NoOpNoOp^Variable/Assign
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
a
b
c
	asset
classify_x2y3
classify_xy
predict
regress_x2y3

regress_xy	
regress_xy2


signatures*
71
VARIABLE_VALUEaa/.ATTRIBUTES/VARIABLE_VALUE*
71
VARIABLE_VALUEbb/.ATTRIBUTES/VARIABLE_VALUE*
71
VARIABLE_VALUEcc/.ATTRIBUTES/VARIABLE_VALUE*
* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 
�
regress_x_to_y
regress_x_to_y2
regress_x2_to_y3
classify_x_to_y
classify_x2_to_y3
serving_default* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_6StatefulPartitionedCallsaver_filenameabcConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto� 82J 

CPU

GPU *%
f R
__inference__traced_save_300
�
StatefulPartitionedCall_7StatefulPartitionedCallsaver_filenameabc*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto� 82J 

CPU

GPU *(
f#R!
__inference__traced_restore_318��
�:B >

_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0S
MulMulMul/ReadVariableOp:value:0inputs*
T0*
_output_shapes
:f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0V
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*
_output_shapes
:I
IdentityIdentityAdd:z:0^NoOp*
T0*
_output_shapes
:L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_regress_x2y3_165

inputs%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB `
ParseExample/key_x2Const*
_output_shapes
:*
dtype0*
valueB*    d
ParseExample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
ParseExample/ReshapeReshapeParseExample/key_x2:output:0#ParseExample/Reshape/shape:output:0*
T0*
_output_shapes
:d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB t
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBxBx2j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2inputs*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Reshape:output:0*
Tdense
2*:
_output_shapes(
&:���������:���������*
dense_shapes
::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0�
MulMulMul/ReadVariableOp:value:0*ParseExample/ParseExampleV2:dense_values:0*
T0*'
_output_shapes
:���������f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0c
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
IdentityIdentityAdd:z:0^NoOp*
T0*'
_output_shapes
:���������L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_regress_xy2_145

inputs%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�:B >

_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0S
MulMulMul/ReadVariableOp:value:0inputs*
T0*
_output_shapes
:f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0V
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*
_output_shapes
:I
IdentityIdentityAdd:z:0^NoOp*
T0*
_output_shapes
:L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_classify_x2y3_215

inputs%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�$:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_user_specified_namea:!

_user_specified_nameb:!

_user_specified_namec:=9

_output_shapes
: 

_user_specified_nameConst2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : "!

identity_7Identity_7:output:0w
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: j
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_a"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_a^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: n
Read_1/DisableCopyOnReadDisableCopyOnReadread_1_disablecopyonread_b"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOpread_1_disablecopyonread_b^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: n
Read_2/DisableCopyOnReadDisableCopyOnReadread_2_disablecopyonread_c"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpread_2_disablecopyonread_c^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�Ba/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEBc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_6Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_7IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp*
_output_shapes
 
�
__inference__traced_save_300
file_prefix"
read_disablecopyonread_a: $
read_1_disablecopyonread_b: $
read_2_disablecopyonread_c: 
savev2_const

identity_7��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp
�:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_user_specified_namea:!

_user_specified_nameb:!

_user_specified_namec2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : "!

identity_4Identity_4:output:0�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�Ba/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEBc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_aIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_bIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_cIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: a
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
_output_shapes
 
�
__inference__traced_restore_318
file_prefix
assignvariableop_a: 
assignvariableop_1_b: 
assignvariableop_2_c: 

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB `
ParseExample/key_x2Const*
_output_shapes
:*
dtype0*
valueB*    d
ParseExample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
ParseExample/ReshapeReshapeParseExample/key_x2:output:0#ParseExample/Reshape/shape:output:0*
T0*
_output_shapes
:d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB t
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBxBx2j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2inputs*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Reshape:output:0*
Tdense
2*:
_output_shapes(
&:���������:���������*
dense_shapes
::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0�
MulMulMul/ReadVariableOp:value:0*ParseExample/ParseExampleV2:dense_values:0*
T0*'
_output_shapes
:���������f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0c
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
IdentityIdentityAdd:z:0^NoOp*
T0*'
_output_shapes
:���������L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_regress_xy_115

inputs%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�:= 9

_output_shapes
:

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0N
MulMulMul/ReadVariableOp:value:0x*
T0*
_output_shapes
:f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0V
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*
_output_shapes
:I
IdentityIdentityAdd:z:0^NoOp*
T0*
_output_shapes
:L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_predict_235
x%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:#

_user_specified_name119:#

_user_specified_name12122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *#
fR
__inference_regress_xy_115o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
,__inference_signature_wrapper_regress_xy_125

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:#

_user_specified_name199:#

_user_specified_name20122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *$
fR
__inference_classify_xy_195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
-__inference_signature_wrapper_classify_xy_205

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall
�:B >

_output_shapes
:
 
_user_specified_nameinputs:#

_user_specified_name219:#

_user_specified_name22122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *&
f!R
__inference_classify_x2y3_215b
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
/__inference_signature_wrapper_classify_x2y3_225

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB `
ParseExample/key_x2Const*
_output_shapes
:*
dtype0*
valueB*    d
ParseExample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
ParseExample/ReshapeReshapeParseExample/key_x2:output:0#ParseExample/Reshape/shape:output:0*
T0*
_output_shapes
:d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB t
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBxBx2j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2inputs*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Reshape:output:0*
Tdense
2*:
_output_shapes(
&:���������:���������*
dense_shapes
::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 f
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0�
MulMulMul/ReadVariableOp:value:0*ParseExample/ParseExampleV2:dense_values:0*
T0*'
_output_shapes
:���������f
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0c
AddAddV2Mul:z:0Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
IdentityIdentityAdd:z:0^NoOp*
T0*'
_output_shapes
:���������L
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp*
_output_shapes
 
�
__inference_classify_xy_195

inputs%
mul_readvariableop_resource: %
add_readvariableop_resource: 
identity��Add/ReadVariableOp�Mul/ReadVariableOp
�:B >

_output_shapes
:
 
_user_specified_nameinputs:#

_user_specified_name169:#

_user_specified_name17122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *%
f R
__inference_regress_x2y3_165b
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
.__inference_signature_wrapper_regress_x2y3_175

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall
�:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:#

_user_specified_name149:#

_user_specified_name15122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU *$
fR
__inference_regress_xy2_145o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
-__inference_signature_wrapper_regress_xy2_155

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall
�:= 9

_output_shapes
:

_user_specified_namex:#

_user_specified_name239:#

_user_specified_name24122
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes

:: : "
identityIdentity:output:0�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto� 82J 

CPU

GPU * 
fR
__inference_predict_235b
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
)__inference_signature_wrapper_predict_245
x
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall"�
�82unknown*2.14.0"serve�
D
AddV2
x"T
y"T
z"T":
2	type
T��
^
AssignVariableOp
resource
value"dtype"type
dtype"( bool
validate_shape�
8
Const
output"dtype"tensor
value"type
dtype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	type
T
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"(bool
delete_old_dirs"( bool
allow_missing_files�
?
Mul
x"T
y"T
z"T":
2	type
T�

NoOp
M
Pack
values"T*N
output"T"0(int
N"	type
T" int
axis
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types":
2	(
list(type)
Tdense"(int

num_sparse"%:
2	(
list(type)
sparse_types"+:
2	(
list(type)
ragged_value_types"*:
2	(
list(type)
ragged_split_types"(list(shape)
dense_shapes
C
Placeholder
output"dtype"type
dtype":shape
shape
X
PlaceholderWithDefault
input"dtype
output"dtype"type
dtype"shape
shape
@
ReadVariableOp
resource
value"dtype"type
dtype�
[
Reshape
tensor"T
shape"Tshape
output"T"	type
T":
2	0type
Tshape
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"0(
list(type)
dtypes�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"0(
list(type)
dtypes�
?
Select
	condition

t"T
e"T
output"T"	type
T
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"(
list(type)
Tin"(
list(type)
Tout"	func
f" string
config" string
config_proto" string
executor_type��
@
StaticRegexFullMatch	
input

output
"string
pattern
L

StringJoin
inputs*N

output"
(int
N" string
	separator
�
VarHandleOp
resource" string
	container" string
shared_name"type
dtype"shape
shape"#
 list(string)
allowed_devices�
9
VarIsInitializedOp
resource
is_initialized
�