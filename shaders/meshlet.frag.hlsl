struct PSInput {
  [[vk::location(0)]] float3 normal : NORMAL;
  [[vk::location(1)]] float4 color : COLOR;
};

static const float3 lightDir = normalize(float3(0.4, 0.8, 0.6));

float4 main(PSInput input) : SV_Target0 {
  float lighting =
      max(dot(normalize(input.normal), lightDir), 0.0) * 0.5 + 0.01;
  float3 base = input.color.rgb;
  return float4(base * lighting, 1.0);
}
