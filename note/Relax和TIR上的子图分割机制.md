# 再聊Relax上的子图分割机制

## Relax子图分割机制

Relax IR中的子图分割机制主要基于**模式匹配**和**变换pass**来实现，核心组件是`FuseOpsByPattern`变换。

### 基于模式的子图识别

Relax使用模式注册系统来定义可分割的操作序列。以cuBLAS后端为例，系统注册了多种矩阵乘法相关的模式，包括基础矩阵乘法、带偏置的矩阵乘法、以及融合激活函数的模式。

```python
register_patterns(
    [
        (
            "cublas.matmul",
            *make_matmul_pattern(
                with_bias=False,
            ),
            _check_matmul,
        ),
        (
            "cublas.matmul_bias",
            *make_matmul_pattern(
                with_bias=True,
            ),
            _check_matmul,
        ),
        (
            "cublas.matmul_bias_relu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
            _check_matmul,
        ),
        (
            "cublas.matmul_bias_gelu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
            ),
            _check_matmul,
        ),
```

每个模式都配有相应的检查函数来验证操作是否满足特定后端的要求，如数据类型支持、张量形状约束等。 

```python
def _check_matmul(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]
    matmul_call = context.annotated_expr["root"]

    if "scale" in context.annotated_expr and "zp" in context.annotated_expr:
        scale = context.annotated_expr["scale"]
        zero_point = context.annotated_expr["zp"]
        # Only scalar values for scale and zero_point are supported.
        if scale.struct_info.ndim != 0 or zero_point.struct_info.ndim != 0:
            return False
        # Only zero_point == 0.0 is supported.
        if zero_point.data.numpy()[()].item() != 0.0:
            return False

    lhs_dtype = lhs.struct_info.dtype
    rhs_dtype = rhs.struct_info.dtype
    out_dtype = matmul_call.struct_info.dtype
    if not _is_supported_dtype(lhs_dtype, rhs_dtype, out_dtype):
        return False

    lhs_shape = lhs.struct_info.shape.values
    rhs_shape = rhs.struct_info.shape.values

    if not isinstance(lhs_shape[-1], (tvm.tir.expr.IntImm, int)):
        # Reduction axis must be constant
        return False

    if lhs_dtype == "int8" and rhs_dtype == "int8":
        if lhs_shape[-1] % 4 != 0:
            # Reduction axis must be multiples of 4 for IGEMM
            return False
        if not isinstance(rhs_shape[-1], (tvm.tir.expr.IntImm, int)) or rhs_shape[-1] % 4 != 0:
            # Rows number must be multiples of 4 for IGEMM
            return False
```

1. PatternCheckContextNode (include/tvm/relax/transform.h:426) -实际的数据节点类，包含以下成员：

​	  \- matched_expr: 与 FusionPattern 匹配的表达式

​	  \- annotated_expr: 这是一个字典，用于存储模式匹配过程中标识符到匹配表达式的映射。

​	  \- matched_bindings: 这是一个字典，用于存储变量到其绑定表达式的映射

​	  \- var_usages: 变量定义到使用点的映射

​          - value_to_bound_var: 值到绑定变量的映射

```c++
class PatternCheckContextNode : public Object {
 public:
  /*!
   * \brief The expression that's matched with the FusionPattern::pattern.
   */
  Expr matched_expr;

  /*!
   * \brief A map which contains all expressions matched by the sub patterns in
   * FusionPattern::annotation_patterns.
   */
  Map<String, Expr> annotated_expr;

  /*!
   * \brief Map from variable to its value. It contains variables from bindings that
   * is being fused by FuseOpsByPattern.
   */
  Map<Var, Expr> matched_bindings;

  /*!
   * \brief A map mapping variable definitions to a set of uses. It has all variables
   * used in the function.
   */
  Map<Var, Array<Var>> var_usages;

  /*!
   * \brief Map from value to its bound variable. It doesn't have variables after the
   * matched expression.
   */
  Map<Expr, Var> value_to_bound_var;
```

2. PatternCheckContext (include/tvm/relax/transform.h:471) - 对 PatternCheckContextNode的引用包装类

 这个数据类型主要用于 Relax 中的算子融合优化，作为 FusionPattern::check函数的输入，提供模式匹配所需的上下文信息。

```c++
class PatternCheckContext : public ObjectRef {
 public:
  PatternCheckContext(Expr matched_expr, Map<String, Expr> annotated_expr,
                      Map<Var, Expr> matched_bindings, Map<Var, Array<Var>> var_usages,
                      Map<Expr, Var> value_to_bound_var);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PatternCheckContext, ObjectRef,
                                            PatternCheckContextNode);
};
```

3. PatternCheckContext 的实现代码在 src/relax/transform/fuse_ops.cc:1414 中。实现很简单：

- 构造函数 (src/relax/transform/fuse_ops.cc:1414-1425) - 创建一个 PatternCheckContextNode 对象并初始化所有成员变量
- 节点注册 (src/relax/transform/fuse_ops.cc:1427) - 通过 TVM_REGISTER_NODE_TYPE 注册到TVM 的对象系统中

```c++
PatternCheckContext::PatternCheckContext(Expr matched_expr, Map<String, Expr> annotated_expr,
                                         Map<Var, Expr> matched_bindings,
                                         Map<Var, Array<Var>> var_usages,
                                         Map<Expr, Var> value_to_bound_var) {
  ObjectPtr<PatternCheckContextNode> n = make_object<PatternCheckContextNode>();
  n->matched_expr = std::move(matched_expr);
  n->annotated_expr = std::move(annotated_expr);
  n->matched_bindings = std::move(matched_bindings);
  n->var_usages = std::move(var_usages);
  n->value_to_bound_var = std::move(value_to_bound_var);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternCheckContextNode);
```

 在同一文件中，CreatePatternCheckContext 方法 (src/relax/transform/fuse_ops.cc:1191-1209)展示了如何创建这个上下文对象，它在模式匹配过程中收集必要的信息来构建 PatternCheckContext实例。

```c++
  PatternCheckContext CreatePatternCheckContext(const CallNode* call,
                                                const Map<DFPattern, Expr>& matched_result) {
    Map<String, Expr> annotated_expr;
    for (const auto& it : annotation_pat_) {
      if (matched_result.count(it.second)) {
        annotated_expr.Set(it.first, matched_result[it.second]);
      }
    }

    Map<Var, Expr> matched_bindings;
    for (const auto& [pat, match] : matched_result) {
      if (pat->IsInstance<CallPatternNode>() || pat->IsInstance<TupleGetItemPatternNode>()) {
        matched_bindings.Set(value_to_bound_var_[match], match);
      }
    }

    return PatternCheckContext(GetRef<Call>(call), annotated_expr, matched_bindings,
                               current_block_use_def_, value_to_bound_var_);
  }
```

`PatternCheckContext` 是 TVM 模式匹配系统中的核心数据结构，它为模式检查函数提供了访问匹配表达式和变量绑定的接口。通过 `annotated_expr` 可以获取模式中命名的子表达式，通过 `matched_bindings` 可以获取变量的具体绑定信息，这使得模式检查函数能够对匹配的子图进行详细的验证和约束检查。

### 分割过程实现

子图分割通过`partition_for_cublas`等函数实现，该函数使用`FuseOpsByPattern`变换，传入注册的模式并设置相应的标注参数。 cublas.py:222-244



```python
def partition_for_cublas(mod, bind_constants=False):
    """
    Partition the input module into cuBLAS-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    bind_constants : bool
        Whether or not to keep bound constants in the grouped function.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the cuBLAS backend.
    """

    patterns = get_patterns_with_prefix("cublas")
    return transform.FuseOpsByPattern(
        patterns, bind_constants=bind_constants, annotate_codegen=True
    )(mod)
```

### 复合函数的合并

分割后的子图通过`MergeCompositeFunctions`变换进行进一步处理。该变换将多个标记为"Composite"的函数合并成更大的、标记为"Codegen"的函数，以实现更高效的后端offloading。

```python
@tvm.script.ir_module
class Conv2dReLUx2_merged:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dReLUx2_merged
        with R.dataflow():
            gv: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu_relax_nn_conv2d_relax_nn_relu1_dnnl(
                data, weight1, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_relax_nn_conv2d_relax_nn_relu1_dnnl(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Codegen": "dnnl"})

        @R.function
        def lv(
            data11: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight111: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "dnnl.conv2d_relu"})
            with R.dataflow():
                lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data11,
                    weight111,
                    padding=[1, 1, 1, 1],
                )
                gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
                R.output(gv1)
            return gv1

        lv2: R.Tensor((1, 64, 56, 56), dtype="float32") = lv(data1, weight11)

        @R.function
        def lv11(
            conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight211: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "dnnl.conv2d_relu"})
            with R.dataflow():
                lv21: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                    conv1,
                    weight211,
                    padding=[0, 0, 0, 0],
                )
                gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv21)
                R.output(gv2)
            return gv2

        gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = lv11(lv2, weight21)
        return gv3
```

FuseOpsByPattern 是 TVM Relax 中的一个编译器  Pass，它不是一个数据结构，而是一个函数/变换。相关的核心数据结构是 FusionPattern：

  FusionPattern 数据结构包含以下字段：

  1. name (String) - 模式名称，匹配成功后会作为融合函数的 kComposite 属性值
  2. pattern (DFPattern) - 数据流模式，用于在 DataflowBlock 中匹配表达式。被模式覆盖的所有 call
    节点会被提取到融合函数中
  3. annotation_patterns (Map<String, DFPattern>) -
    用于从模式匹配结果中提取重要表达式的映射。这个映射中的所有 DFPattern 都必须是 pattern 的一部分
  4. check (Optionalffi::Function) - 可选的检查函数，用于决定是否接受匹配结果。签名为 bool(const         
    PatternCheckContext& context)
  5. attrs_getter (Optionalffi::Function) - 可选的函数，用于获取融合函数的属性。签名为 Map<String,       
    Any>(const Map<String, Expr>& context)

  FuseOpsByPattern 的作用：
  - 对模块中的每个函数应用模式匹配
  - 将匹配到的表达式分组到新函数中
  - 结果类似于 FuseOps，但融合完全由提供的模式驱动
  - 只在数据流块内操作

### MergeCompositeFunctions

#### 声明

```c++
TVM_DLL Pass MergeCompositeFunctions();

/*!
 * \brief Fuse relax sub-function into a larger TIR function if possible.
    this pass works together with FuseOps to perform operator fusion.

 * \return The Pass.
 */
```

该函数通过 FFI 机制导出到 Python 接口。

```C++
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.MergeCompositeFunctions", MergeCompositeFunctions);
});
```

在 Python 层面，该 transform 被导入到 `relax.transform` 模块中。 __init__.py:60

```python
    LiftTransformParams,
    LowerAllocTensor,
    LowerRuntimeBuiltin,
    MergeCompositeFunctions,
    MetaScheduleApplyDatabase,
    MetaScheduleTuneIRMod,
    MetaScheduleTuneTIR,
```

#### 实现

  实现结构：

  1. 主要类：

    - CompositeGroupsBuilder (src/relax/transform/merge_composite_functions.cc:75) - 构建组合函数分组
    - CompositeInliner (src/relax/transform/merge_composite_functions.cc:305) - 内联组合函数
    - CompositeFunctionAnnotator (src/relax/transform/merge_composite_functions.cc:345) - 注解组合函数
  2. 核心功能 (src/relax/transform/merge_composite_functions.cc:402-411)：

    - 将 FuseOpsByPattern 创建的多个组合函数合并成一个新函数
    - 新函数会被标注 kCodegen 和 kGlobalSymbol 属性
    - 用于将计算卸载到外部后端
  3. 算法流程：

    - 使用 CompositeGroupsBuilder 分析数据流并创建分组
    - 检查是否可以将子组合并到父组（需要相同的后端且不创建循环依赖）
    - 通过 MakeGroupedFunctions 创建新的分组函数
    - 使用 CompositeFunctionAnnotator 添加必要的注解
    - 最后执行死代码消除
  4. Python 接口 (python/tvm/relax/transform/transform.py:961)：

    - 简单封装 C++ 实现，无额外参数

##### 核心实现函数

主要的实现逻辑在 `MergeCompositeFunctions(IRModule mod)` 函数中，该函数执行整个合并过程。 merge_composite_functions.cc:402-412（对应上面3的算法流程）

```c++
IRModule MergeCompositeFunctions(IRModule mod) {
  auto gvar = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(gvar));
  support::Arena arena;
  auto group_map = CompositeGroupsBuilder(mod, &arena).Run(func);
  auto new_mod = MakeGroupedFunctions(mod, group_map);
  new_mod = CompositeFunctionAnnotator(mod, new_mod).update();

  // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
  return DeadCodeElimination(new_mod, {"main"});
}
```

##### 关键组件

###### CompositeGroupsBuilder 类

负责根据数据流为每个子表达式分配组，并返回从子表达式到其组的映射。 merge_composite_functions.cc:75-105

```c++
class CompositeGroupsBuilder : public MemoizedExprTranslator<Group*> {
 public:
  using GroupMap = std::unordered_map<const Object*, Group*>;
  using MemoizedExprTranslator<Group*>::VisitExpr_;

  CompositeGroupsBuilder(IRModule mod, support::Arena* arena) : mod_(mod), arena_(arena) {}

  GroupMap Run(Function func) {
    for (const auto& param : func->params) {
      memo_[param] = arena_->make<Group>();
    }

    PostOrderVisit(func, [this](Expr e) {
      // Make default groups for dataflow nodes other than CallNode.
      // Groups for CallNode are created in its visitor.
      if (e->IsInstance<ConstantNode>() || e->IsInstance<ShapeExprNode>() ||
          e->IsInstance<TupleNode>() || e->IsInstance<TupleGetItemNode>() ||
          e->IsInstance<PrimValueNode>()) {
        memo_[e] = arena_->make<Group>();
      }
    });

    VisitExpr(func->body);

    GroupMap group_map;
    for (const auto& [expr, group] : memo_) {
      group_map[expr.get()] = group->FindRoot();
    }

    return group_map;
  }
```

该类的核心算法在 `VisitExpr_` 方法中处理 `CallNode`，决定是否可以合并组。 merge_composite_functions.cc:144-166

```c++
  Group* VisitExpr_(const CallNode* call) {
    std::vector<Group*> groups_to_merge = GetGroupsToMerge(call);
    Group* group;

    if (groups_to_merge.size() == 0) {
      // Create new group if there is nothing to merge with
      group = CreateNewGroup(call);
    } else {
      auto it = groups_to_merge.cbegin();
      // Assign the first mergable group to current node
      // to reduce the number of groups created
      group = *it++;
      group->num_nodes += 1;

      // Merge all groups
      for (; it != groups_to_merge.cend(); ++it) {
        MergeGroup(*it, group);
      }
    }

    UpdateGroupDependencies(group, call->args);
    return group;
  }
```

**MergeGroup 的概念解读**

核心功能

  MergeGroup 是在 merge_composite_functions.cc 中用于将多个复合函数（composite     
  functions）合并成一个更大函数的核心机制。这个过程主要用于外部后端的代码生成优化。

  工作原理

  1. Group 结构：

    - 继承自 GraphPartitioner::Group，使用并查集（Union-Find）数据结构
    - 每个 Group 代表一组可以合并的算子或函数
    - 包含属性：父节点、节点数量、codegen 名称等
  2. MergeGroup 函数的实现 (src/relax/transform/merge_composite_functions.cc:199-221)：
        
      ```c++
        void MergeGroup(Group* from, Group* to) {
        // 1. 确保两个组的 codegen 目标相同
            ICHECK_EQ(GetCodegenName(from), GetCodegenName(to));
      
            // 2. 找到各自的根节点
        Group* from_root = from->FindRoot();
            Group* to_root = to->FindRoot();
      
            // 3. 执行合并：将 from 的根指向 to 的根
        from_root->parent = to_root;
            to_root->num_nodes += from_root->num_nodes;
      
            // 4. 更新依赖关系
        group_deps_[to_root].merge(group_deps_[from_root]);
        // 维护所有组都是根组的不变量

    }
  ```
  
  
  

  ```

 主要方法

1. Run 方法 (src/relax/transform/merge_composite_functions.cc:310-316):

```c++
  Function Run(Function func) {
      inlined_functions_ = Map<Function, Function>();
      auto new_body = VisitExpr(ToNonDataflow(func->body));
      auto new_func = Function(func->params, new_body, func->ret_struct_info,
                             func->is_pure, func->attrs, func->span);
      return new_func;
  }
```

  - 初始化内联函数缓存
  - 将函数体转换为非数据流形式后进行访问
  - 创建新函数并返回

2. VisitExpr_ 方法 (src/relax/transform/merge_composite_functions.cc:318-333):

```C++
  Expr VisitExpr_(const CallNode* call) {
      if (call->op->IsInstance<GlobalVarNode>()) {
          auto gvar = Downcast<GlobalVar>(call->op);
          auto func = Downcast<Function>(mod_->Lookup(gvar));

          // 检查是否是复合函数
          if (func->GetAttr<String>(attr::kComposite)) {
              // 如果还未内联过，创建新的函数副本
              if (!inlined_functions_.count(func)) {
                  auto new_func = CopyWithNewVars(func);
                  // 移除 kPrimitive 属性
                  new_func = WithoutAttr(new_func, tvm::relax::attr::kPrimitive);
                  inlined_functions_.Set(func, new_func);
              }
              // 返回对内联函数的调用
              return Call(inlined_functions_[func], call->args);
          }
      }
      return ExprMutator::VisitExpr_(call);
  }
```

  合并条件

  文件开头的注释清楚地说明了合并条件：
  1. 相同的后端目标：参数必须是调用到相同后端的复合函数的结果
  2. 无循环依赖：合并不会创建与其他父组的循环依赖

  算法流程

  1. 构建组：CompositeGroupsBuilder 遍历函数，为每个子表达式分配组
  2. 检查可合并性：GetGroupsToMerge 检查哪些组可以合并
  3. 执行合并：调用 MergeGroup 将符合条件的组合并
  4. 更新依赖：UpdateGroupDependencies 维护组间的依赖关系

  依赖关系管理

  // group_deps_ 映射：组 -> 其依赖的组集合
  std::unordered_map<Group*, std::unordered_set<Group*>> group_deps_;

  这个数据结构用于：
  - 跟踪组之间的直接和间接依赖
  - 检测潜在的循环依赖
  - 在合并时传播依赖关系

  实际应用

  这个机制主要用于：
  1. 优化外部后端编译：将多个小的复合函数合并成大函数，减少函数调用开销
  2. 自包含函数生成：确保生成的函数包含所有必要的计算，不需要引用原始模块
  3. 避免循环依赖：通过依赖分析确保合并后的图仍然是有向无环图（DAG）

  示例场景

  文档中给出的例子展示了为什么需要依赖检查：
  O = Offloaded to A
  X = Offloaded to B

      O         O
     / \       /
    O   X --> 不能将底部的 O 合并到左边的父组
     \ /      因为右边的 X 依赖于左边组的输出
      O

  这个机制确保了代码生成的正确性和效率，是 TVM 中实现高效外部后端集成的重要组成部分。

###### CompositeInliner 类

核心功能

CompositeInliner 的主要作用是将全局级别的复合函数（composite  functions）内联到它们的调用点。这是为了让 MergeCompositeFunctions创建的函数成为自包含的，使得每个外部后端编译器不需要引用原始的包含模块。merge_composite_functions.cc:305-316

类定义分析

```C++
class CompositeInliner : public ExprMutator {
   public:
    explicit CompositeInliner(IRModule mod) : ExprMutator(mod), mod_(mod) {}

   private:
    IRModule mod_;
    Map<Function, Function> inlined_functions_;  // 缓存已内联的函数
  };
```

 工作流程

  1. 识别复合函数调用：

    - 检查调用节点是否调用全局变量
    - 查找该全局变量对应的函数
    - 检查函数是否有 kComposite 属性
  2. 内联处理：

    - 对于复合函数，创建一个新的函数副本（使用 CopyWithNewVars）
    - 移除 kPrimitive 属性（因为内联后不再是原始操作）
    - 缓存内联的函数以避免重复创建
  3. 替换调用：

    - 将原始的全局函数调用替换为对内联函数的直接调用
    - 保持相同的参数

###### CompositeFunctionAnnotator 类

核心功能

  CompositeFunctionAnnotator 的主要作用是为每个创建的复合函数包装一个外层函数，这个外层函数的
  函数体只包含对复合函数的调用，并且为外层函数添加 kCodegen 和 kGlobalSymbol 属性标注。      

类定义分析

```c++
  class CompositeFunctionAnnotator : public ExprMutator {
   public:
    explicit CompositeFunctionAnnotator(IRModule mod, IRModule new_mod)
        : ExprMutator(new_mod), mod_(new_mod), inliner(mod) {
      mod_.CopyOnWrite();
    }

   private:
    IRModule mod_;                                        // 正在处理的模块
    CompositeInliner inliner;                            // 内联器实例
    std::unordered_map<GlobalVar, GlobalVar> var_map_;   // 旧变量到新变量的映射
  };
```

主要方法

update 方法 (行353-358):

```c++
  IRModule update() {
    auto gvar = mod_->GetGlobalVar("main");
    auto func = Downcast<Function>(mod_->Lookup(gvar));
    builder_->UpdateFunction(gvar, Downcast<Function>(VisitExpr(func)));
    return builder_->GetContextIRModule();
  }
```

  - 获取 main 函数
  - 访问并转换函数
  - 更新模块中的函数
  - 返回更新后的模块

-VisitExpr_ 方法 (行360-392):

  这是核心处理逻辑，对每个函数调用进行处理：

```
  Expr VisitExpr_(const CallNode* call) {
      if (call->op->IsInstance<GlobalVarNode>()) {
          GlobalVar cur_var = Downcast<GlobalVar>(call->op);
          auto func = Downcast<Function>(mod_->Lookup(cur_var));

          // 检查是否有 kCodegen 属性
          if (auto codegen_name = func->GetAttr<String>(attr::kCodegen)) {
              GlobalVar new_var;

              // 检查是否已经处理过这个函数
              if (var_map_.count(cur_var) > 0) {
                  new_var = var_map_[cur_var];
              } else {
                  // 第一次处理，创建新函数
                  // 1. 从模块中移除旧函数
                  auto old_var =
  builder_->GetContextIRModule()->GetGlobalVar(cur_var->name_hint);
                  builder_->GetContextIRModule()->Remove(old_var);

                  // 2. 重命名函数
                  String new_func_name = cur_var->name_hint + "_" + codegen_name.value();

                  // 3. 内联复合函数
                  Function new_func = inliner.Run(Downcast<Function>(func));

                  // 4. 添加属性
                  new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol, new_func_name);
                  new_func = WithoutAttr(std::move(new_func), tvm::relax::attr::kPrimitive);       

                  // 5. 添加新函数到模块
                  new_var = builder_->AddFunction(new_func, new_func_name);
                  var_map_[cur_var] = new_var;
              }

              // 返回对新函数的调用
              return Call(new_var, call->args);
          }
      }
      return GetRef<Call>(call);
  }
```

 工作流程

  1. 函数识别：

    - 遍历所有函数调用
    - 检查被调用的函数是否有 kCodegen 属性
  2. 函数转换：

    - 使用 CompositeInliner 内联复合函数
    - 生成新的函数名：原函数名_codegen名称
    - 添加 kGlobalSymbol 属性（用于外部链接）
    - 移除 kPrimitive 属性
  3. 缓存管理：

    - 使用 var_map_ 缓存已处理的函数
    - 避免重复处理相同的函数

  设计目的

  1. 外部后端集成：

    - 生成的函数带有 kGlobalSymbol，可以被外部编译器识别
    - 函数名包含 codegen 信息，便于区分不同后端
  2. 自包含性：

    - 通过 CompositeInliner 确保函数包含所有必要的逻辑
    - 不依赖模块中的其他全局函数
  3. 模块化：

    - 每个 codegen 后端的函数都被独立处理
    - 保持原始调用接口不变（参数相同）

 在 MergeCompositeFunctions 的处理流程中：

    1. CompositeGroupsBuilder 分析并构建组
    2. MakeGroupedFunctions 根据分组创建函数
    3. CompositeFunctionAnnotator 为这些函数添加必要的属性和重命名
    4. 最终生成适合外部后端编译的独立函数

 这个类是实现 TVM与外部编译器集成的关键组件，确保生成的函数符合外部编译器的要求，并且可以独立编译和执行。

##### 小结

`MergeCompositeFunctions` 是 TVM Relax 中一个重要的 transformation pass，它通过智能分组和合并具有相同 codegen 属性的 composite functions，为外部后端编译器的代码生成奠定基础。该实现考虑了复杂的数据流依赖关系，确保合并后的函数在语义上正确且可以被外部编译器处理。

## TIR子图分割机制

TIR层面的子图分割主要通过**调度系统(Schedule System)**实现，与Relax层面的模式匹配不同，TIR更关注底层的计算块操作和优化。严格意义上来讲，TIR上不存在子图分割：**Subgraph partitioning does not exist as a direct feature at the TIR level in TVM.** Instead, subgraph partitioning primarily occurs at higher levels of the compilation stack, particularly in Relax IR.

### 调度原语操作

TIR调度系统提供了丰富的原语操作如`compute_at`、`compute_inline`等，这些操作可以改变计算块在IR中的位置和执行方式，从而实现子图级别的优化。这些原语操作是TIR子图分割和重组的基础工具。

## 核心特点对比

1. **Relax层面**：侧重于高层次的算子融合和后端分发，通过模式匹配识别可优化的操作序列
2. **TIR层面**：侧重于底层的张量计算优化，通过调度变换改变计算的执行策略

两个层面的子图分割机制相互配合，形成了TVM完整的子图优化体系，从高层的算子融合到底层的内存和计算优化。

## 总结

Relax和TIR的子图分割机制体现了TVM分层优化的设计理念。Relax层面的分割主要服务于算子融合和后端dispatch，而TIR层面的分割则更多关注于具体的计算优化。这种分层设计使得TVM能够在不同抽象层次上进行灵活的优化，既支持高层的模式识别和后端offloading，也支持底层的细粒度调度优化。

# Group

Based on my analysis of the codebase, the `Group` data structure in the context of `src/relax/transform/merge_composite_functions.cc` is defined as a type alias to `GraphPartitioner::Group`.

The type alias is defined as: [1](#2-0) 

The actual data structure definition of `GraphPartitioner::Group` is a struct that serves as a union find data structure for graph partitioning: [2](#2-1) 

```c++
  struct Group {
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief reference to the root node. */
    const tvm::Object* root_ref{nullptr};
    /*!
     * \brief Reference to the anchor node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Object* anchor_ref{nullptr};
    /*!
     * \brief The number of nodes belonging to this group
     */
    uint32_t num_nodes{1};
    /*!
     * \brief The number of function arguments belonging to this group
     */
    size_t args_num{0};

    /*! \brief Optional attributes to annotate the grouped function. */
    Map<String, Any> attrs;
    /*!
     * \brief Find the group root, perform path compression
     * \return The root type node.
     */
    Group* FindRoot();
  };
```

The `Group` struct contains the following key members:
- `parent`: A pointer to the parent in the union find data structure
- `pattern`: The operation pattern kind of the group  
- `root_ref`: Reference to the root node
- `anchor_ref`: Reference to the anchor node (only used for kOutEWiseFusable pattern)
- `num_nodes`: The number of nodes belonging to this group
- `args_num`: The number of function arguments belonging to this group
- `attrs`: Optional attributes to annotate the grouped function
- `FindRoot()`: A method to find the group root with path compression

## 小结

The `Group` struct is designed to support the merge composite functions transformation by organizing nodes into groups that can be fused together. It uses a union-find data structure pattern to efficiently manage group membership and merging operations. The struct is part of the graph partitioning analysis framework used by the Relax IR transformation passes.

# GraphPartitioner类

GraphPartitioner 是 TVM Relax 中用于算子融合（operator fusion）的核心组件。它负责将计算图中的算子分组，以便将多个算子融合成一个函数来提高执行效率。       

  主要组成部分：

  1. IndexedForwardGraph - 索引化的前向数据流图

    - 用于表示算子之间的依赖关系
    - 包含 Node（节点）和 Edge（边）结构
    - 每个节点包含：索引、模式（pattern）、输出边列表等
  2. DominatorTree - 支配树

    - 用于分析节点之间的支配关系
    - 通过 LCA（最低公共祖先）算法计算后支配关系
    - 帮助确定哪些节点可以安全地融合在一起
  3. GraphPartitioner::Group - 分组结构

    - 使用并查集（Union-Find）数据结构管理
    - 每个组包含：父节点、模式、节点数量、参数数量等
    - FindRoot() 方法实现路径压缩优化

  核心算法流程：

  1. 初始化阶段 (InitGroups)：

    - 为每个节点创建独立的组
    - 计算每个节点的参数数量
    - 设置初始模式（pattern）
  2. 融合阶段 (RunFuse)：

    - 分三个阶段执行融合：
      - Phase 0: 处理 OutEWiseFusable 模式（如 conv2d）
      - Phase 1: 处理 Injective 和 Tuple 模式
      - Phase 2: 将 injective 算子融合到中间的 tuple 中
  3. 融合决策条件：

    - 最大融合深度限制 (max_fuse_depth_)
    - 最大函数参数数量限制 (max_function_args_)
    - 算子模式兼容性检查
    - 路径可达性验证

  算子模式 (OpPatternKind)：

  - kElemWise: 逐元素操作
  - kBroadcast: 广播操作
  - kInjective: 单射操作
  - kCommReduce: 交换归约
  - kOutEWiseFusable: 输出可逐元素融合（如卷积）
  - kTuple: 元组操作
  - kOpaque: 不透明操作（不可融合）

  主要优化技术：

  1. 延迟融合 (Postponed Fusing)：

    - 当参数数量未知时延迟融合决策
    - 在后续节点处理时重新评估
  2. 路径检查 (CheckPath)：

    - 确保融合路径上的所有算子都满足模式要求
    - 避免非法融合
  3. 参数计数优化：

    - 动态计算融合后的参数数量
    - 考虑输出参数和动态形状参数

  这个类实现了一个高效的图分割算法，通过智能地将相关算子分组来优化计算图的执行效率，是 TVM
  编译器优化流程中的重要组成部分。

# Relay/analysis目录结构

  1. 基础分析工具

  - analysis.cc: 提供变量分析的基础功能，包括自由变量、绑定变量收集，以及纯函数性检测
  - var2value.cc: 建立变量到值的映射关系，支持常量传播等优化
  - udchain.cc: 构建使用-定义链，分析变量间的依赖关系

  2. 语义检查工具

  - well_formed.cc: 全面的IR合法性检查器，确保：
    - 变量先定义后使用
    - ANF(Administrative Normal Form)形式正确
    - 数据流块约束满足
    - StructInfo完整性
  - struct_info_analysis.cc: 类型系统的核心支持，处理StructInfo和静态类型之间的转换

  3. 优化分析工具

  - graph_partitioner.h/cc: 图分割算法，用于算子融合优化
  - tir_op_pattern_kind.cc: 识别TIR算子的计算模式（逐元素、广播、归约等）
  - computable_at_compile_time.cc: 识别编译时可计算的表达式
  - layout_transformation.cc: 分析内存访问模式，支持数据布局优化

  4. 程序结构分析

  - collect_call_map.cc: 收集函数调用关系，构建调用图
  - detect_recursion.cc: 检测递归调用，使用Tarjan和Johnson算法
  - shape_analysis.cc: 符号形状分析，支持动态形状的等价性判断

  主要应用场景

  1. 编译正确性保证：

    - well_formed检查确保IR合法
    - struct_info_analysis提供类型安全
  2. 优化决策支持：

    - graph_partitioner决定如何融合算子
    - tir_op_pattern_kind帮助识别可融合的模式
    - layout_transformation优化内存访问
  3. 代码生成准备：

    - computable_at_compile_time支持常量折叠
    - var2value支持值传播
    - udchain帮助死代码消除
  4. 模块化编译：

    - collect_call_map支持模块间依赖分析
    - detect_recursion处理递归函数的特殊情况

  这些分析工具共同构成了Relax编译器的分析基础设施，为后续的优化Pass和代码生成提供必要的信息支持。

# 支配树

支配树（Dominator Tree）是编译器优化中的一个重要数据结构，主要用于控制流分析。让我详细解释一下这个概念：

## 支配关系（Dominance）

在控制流图（CFG）中，如果从程序入口到节点 B 的**所有路径**都必须经过节点 A，那么我们说节点 A **支配**（dominates）节点 B，记作 A dom B。

几个关键点：

- 每个节点都支配自己
- 入口节点支配所有节点
- 如果 A 支配 B，且 B 支配 C，则 A 支配 C（传递性）

## 直接支配者（Immediate Dominator）

节点 B 的直接支配者（idom）是指：在所有支配 B 的节点中，离 B 最近的那个节点（除了 B 自己）。每个节点（除了入口节点）都有唯一的直接支配者。

## 支配树的构建

支配树是这样构建的：

- 树的根是 CFG 的入口节点
- 如果节点 A 是节点 B 的直接支配者，那么在支配树中 A 是 B 的父节点

举个例子：

```
CFG:
    1
   / \
  2   3
   \ /
    4
    |
    5
```

在这个 CFG 中：

- 节点 1 支配所有节点
- 节点 4 被 1 支配（所有到 4 的路径都经过 1）
- 节点 5 被 1 和 4 支配

对应的支配树：

```
    1
   /|\
  2 3 4
      |
      5
```

## 支配树的应用

支配树在编译器优化中有多种用途：

1. **SSA 构造**：在构建静态单赋值形式时，需要计算支配边界来确定 φ 函数的插入位置
2. **循环识别**：通过支配关系可以识别自然循环
3. **代码移动**：确定代码可以安全移动的位置
4. **死代码消除**：如果一个定义点支配所有使用点，可以进行某些优化
5. **公共子表达式消除**：在支配关系下可以安全地重用计算结果

## 计算算法

常用的支配树计算算法包括：

- Lengauer-Tarjan 算法：O(n·α(n)) 时间复杂度
- Cooper-Harvey-Kennedy 算法：实践中表现良好的简单算法

支配树是理解和实现许多编译器优化的基础，它提供了程序控制流的层次结构视图，使得许多复杂的优化变得可行。