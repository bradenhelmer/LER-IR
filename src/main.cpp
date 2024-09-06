// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <ler-ir/CommonUtils.h>
#include <ler-ir/IR/LERDialect.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>

namespace cl = llvm::cl;
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;
using namespace ler;

static cl::opt<std::string>
    GenTests("gen-tests", cl::desc("Generate LER-IR MLIR test files in the "
                                   "'test' directory of the given path."));
namespace {
// Anonymous method for generating test MLIR.
static void generateTests(const std::string &dir) {

  std::error_code EC;
  static constexpr const char *FNS[4] = {
      "test/cat1-loop-inv-expr.mlir",
      "test/cat2-partial-loop-inv-expr.mlir",
      "test/cat3-loop-inv-loop.mlir",
      "test/cat4-partial-loop-inv-loop.mlir",
  };

  // First remove old files.
  for (int test_file_idx = 0; test_file_idx < 4; ++test_file_idx) {
    llvm::SmallString<128> path(dir);
    path::append(path, FNS[test_file_idx]);
    if (fs::exists(path)) {
      fs::remove(path);
    }
  }

  // Get MLIR necessities.
  mlir::MLIRContext Ctx;
  mlir::OpBuilder builder(&Ctx);
  Ctx.loadDialect<LERDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  auto FT = builder.getFunctionType({}, {});

  // Cat1: Loop Invariant Expression
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for (i = 0; i < 100; i++) {
  //     result = a * b;
  //     x[i] = result + y[i];
  // }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  auto C1F = mlir::func::FuncOp::create(builder.getUnknownLoc(), "cat1", FT);
  builder.setInsertionPointToStart(C1F.addEntryBlock());
  auto a = QuickConstOp(builder, 20), b = QuickConstOp(builder, 30);
  auto x = builder.create<mlir::memref::AllocOp>(
      builder.getUnknownLoc(),
      mlir::MemRefType::get({100}, mlir::IntegerType::get(&Ctx, 64)));
  auto y = x.clone();
  builder.insert(y);
  auto C1For1 = builder.create<mlir::affine::AffineForOp>(
      builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(C1For1.getBody());
  auto result =
      builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(), a, b);
  auto y_acc = builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), y,
                                                    C1For1.getInductionVar());
  auto x_val_to_store = builder.create<mlir::arith::AddIOp>(
      builder.getUnknownLoc(), result, y_acc);
  auto x_store = builder.create<mlir::memref::StoreOp>(
      builder.getUnknownLoc(), x_val_to_store, x, C1For1.getInductionVar());
  llvm::SmallString<128> path(dir);
  path::append(path, FNS[0]);
  llvm::raw_fd_stream C1O(path, EC);
  C1F.print(C1O);
  path.clear();

  // Cat2: Partialy Loop Invariant Expression
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for(i = 2; i <= M; i++) {
  //	x[i] = y[i-2]+y[i-1]+y[i+1]+y[i+2]
  // }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Cat3: Loop Invariant Loop
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for (i = 0; i <= M; i ++) {
  //     for (j = 0; j <= M; j++) {
  //         for (k = 0; k <= N; k++) {
  //             for(l = 0; l <= N; l++) {
  //                 r[i,k] += x[i,l] * y[l,j] * s[j,k];
  // }}}}
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Cat4: Partialy Loop Invariant Loop
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for(i = 1; i <= M; i++) {
  //     for(j = 1; j <= i; j++) {
  //         y[i] +=x[j];
  // }}
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM(argc, argv);
  llvm::outs() << "Running ler-opt tool...\n";

  if (GenTests != "") {
    generateTests(GenTests);
    return 0;
  }
}
