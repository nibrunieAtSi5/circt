//===- HW.cpp - C Interface for the HW Dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements a C Interface for the HW Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HW.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::hw;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HW, hw, HWDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

int64_t hwGetBitWidth(MlirType type) { return getBitWidth(unwrap(type)); }

bool hwTypeIsAValueType(MlirType type) { return isHWValueType(unwrap(type)); }

bool hwTypeIsAArrayType(MlirType type) { return unwrap(type).isa<ArrayType>(); }

MlirType hwArrayTypeGet(MlirType element, size_t size) {
  return wrap(ArrayType::get(unwrap(element), size));
}

MlirType hwArrayTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ArrayType>().getElementType());
}

intptr_t hwArrayTypeGetSize(MlirType type) {
  return unwrap(type).cast<ArrayType>().getSize();
}

MlirType hwInOutTypeGet(MlirType element) {
  return wrap(InOutType::get(unwrap(element)));
}

MlirType hwInOutTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<InOutType>().getElementType());
}

bool hwTypeIsAInOut(MlirType type) { return unwrap(type).isa<InOutType>(); }

bool hwTypeIsAStructType(MlirType type) {
  return unwrap(type).isa<StructType>();
}

MlirType hwStructTypeGet(MlirContext ctx, intptr_t numElements,
                         HWStructFieldInfo const *elements) {
  SmallVector<StructType::FieldInfo> fieldInfos;
  fieldInfos.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i) {
    auto typeAttr = unwrap(elements[i].attribute).dyn_cast<TypeAttr>();
    if (!typeAttr)
      return MlirType();
    fieldInfos.push_back(
        StructType::FieldInfo{unwrap(elements[i].name), typeAttr.getValue()});
  }
  return wrap(StructType::get(unwrap(ctx), fieldInfos));
}
