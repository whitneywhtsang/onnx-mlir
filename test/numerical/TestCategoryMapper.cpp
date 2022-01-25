/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#define SHARED_LIB_BASE string("./TestCategoryMapper_main_graph")

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

// Test CategoryMapper (with a tensor of int64_t numbers as input).
bool testCategoryMapperInt64ToStr(const int I) {
  MLIRContext ctx;
  setCompileContext(
      ctx, {{OptionKind::CompilerOptLevel, "3"}, {OptionKind::Verbose, "1"}});

  auto loc = UnknownLoc::get(&ctx);
  auto module = ModuleOp::create(loc);
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> shape = {I};
  auto inputType = RankedTensorType::get(shape, builder.getI64Type());
  auto outputType =
      RankedTensorType::get(shape, onnxmlir::StringType::get(&ctx));

  llvm::SmallVector<Type, 1> inputsType{inputType};
  llvm::SmallVector<Type, 1> outputsType{outputType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp = builder.create<FuncOp>(loc, "main_graph", funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto input = entryBlock->getArgument(0);
  auto cats_int64s = builder.getI64ArrayAttr({1, 2, 3});
  auto cats_strings = builder.getStrArrayAttr({"cat", "dog", "human"});
  auto default_string = builder.getStringAttr("unknown");

  auto categoryMapperOp = builder.create<ONNXCategoryMapperOp>(
      loc, outputType, input, cats_int64s, cats_strings, -1, default_string);

  llvm::SmallVector<Value, 1> results = {categoryMapperOp.getResult()};
  builder.create<ReturnOp>(loc, results);
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(loc, funcOp,
      /*numInputs=*/1,
      /*numOutputs=*/1,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);
  if (compileModule(moduleRef, ctx, SHARED_LIB_BASE, onnx_mlir::EmitLib) != 0)
    return false;

  onnx_mlir::ExecutionSession sess(
      getSharedLibName(SHARED_LIB_BASE), "run_main_graph");

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto inputOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>({I}), omTensorDestroy);
  inputs.emplace_back(move(inputOmt));

  return true;
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE));

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestCategoryMapper\n", nullptr, "TEST_ARGS");

  if (!rc::check("Test CategoryMapper with int64_t input vector", []() {
        const int I = *rc::gen::inRange(1, 4);
        RC_ASSERT(testCategoryMapperInt64ToStr(I));
      }))
    return 1;

  return 0;
}
