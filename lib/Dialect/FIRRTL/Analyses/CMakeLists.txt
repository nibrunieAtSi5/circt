add_circt_dialect_library(CIRCTFIRRTLAnalyses
  FanOut.cpp


  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
