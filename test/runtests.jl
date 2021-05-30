using ExBroadcast
using Test
using OffsetArrays, CUDA, StructArrays
@testset "@tab.jl" begin
    a = randn(1000)
    b, c = @tab sincos.(a)
    @test b == sin.(a)
    @test c == cos.(a)
end

@testset "@mtb for OffsetVector" begin
    a = randn(1000)
    aᵒ = OffsetArray(a,-1)
    @mtb b = parent(aᵒ .+ 1)
    @test (a .+ 1) == b
end

@testset "@mtb for BitArray" begin
    a = randn(4096 * 6 - 2048)
    b = a .> 0
    @test @mtb (a .> 0) == b
end

@testset "CUDA supports" begin
    a = CUDA.randn(1000)
    b, c = @tab sincos.(a)
    @test b ≈ sin.(a)
    @test c ≈ cos.(a)
    d = StructArray{ComplexF32}((b, c))
    @test abs.(d) isa CuVector
    e = OffsetArray(d, -1) .+ 1
    @test e isa StructArray
    @test eachindex(e) == 0:999
end
