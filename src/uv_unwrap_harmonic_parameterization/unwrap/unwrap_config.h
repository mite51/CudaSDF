#pragma once

namespace uv {

struct UnwrapConfig {
  int normalClusterK = 12;     
  int minChartTris   = 64;     
  int maxChartTris   = 5000;   
  float maxChartGeodesic = 0.0f; 
  int atlasMaxSize   = 4096;   
  int paddingPx      = 8;      
  bool useCotanWeights = true; 
};

} // namespace uv
