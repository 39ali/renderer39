struct SkyVertexOutput {
  float4 position : SV_Position;
  [[vk::location(0)]] float2 ndc : NDC;
};

// one oversized triangle at the far plane; depth test LESS_OR_EQUAL against
// the cleared depth of 1.0 fills only the pixels no geometry covered.
SkyVertexOutput main(uint vertexIndex : SV_VertexID) {
  const float2 uv = float2((vertexIndex << 1) & 2, vertexIndex & 2);
  SkyVertexOutput output;
  output.ndc = uv * 2.0 - 1.0;
  output.position = float4(output.ndc, 1.0, 1.0);
  return output;
}
