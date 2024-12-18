// Passes.h
// ~~~~~~~~
// LER pass header.
#ifndef LER_PASSES_H
#define LER_PASSES_H

#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERUtils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using llvm::dyn_cast;
using llvm::SmallVector;
using llvm::StringMap;
using llvm::StringRef;
using mlir::BuiltinDialect;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::RewritePatternSet;
using mlir::success;
using mlir::Value;
using mlir::affine::AffineDialect;
using mlir::arith::ArithDialect;
using mlir::arith::IndexCastOp;
using mlir::cf::ControlFlowDialect;
using mlir::func::FuncDialect;
using mlir::memref::AllocOp;
using mlir::memref::LoadOp;
using mlir::memref::MemRefDialect;
using mlir::scf::SCFDialect;

namespace ler {

#define GEN_PASS_DECL
#include <ler-ir/Passes.h.inc>

#define GEN_PASS_REGISTRATION
#include <ler-ir/Passes.h.inc>

} // namespace ler

#define UNKNOWN_LOC ReWriter.getUnknownLoc()

#endif // LER_CONVERSION_PASSES_H
