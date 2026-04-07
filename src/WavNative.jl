module WavNative
using BitIntegers, FixedPointNumbers, SamplesCore
export WavMetadata, get_wav_layout, audioread

struct WavMetadata{T, Channels}
    sample_rate::Int
    data_offset::Int
    total_frames::Int
end

@inline get_transport_type(::Type{T}) where {T} = T === Q0f23 ? Q0f31 : T
const WavMemoryInput = Union{Vector{UInt8},Tuple{Vector{UInt8},Ptr{Cvoid}}}

const RIFF_ID = (UInt8('R'), UInt8('I'), UInt8('F'), UInt8('F'))
const WAVE_ID = (UInt8('W'), UInt8('A'), UInt8('V'), UInt8('E'))
const FMT_ID  = (UInt8('f'), UInt8('m'), UInt8('t'), UInt8(' '))
const DATA_ID = (UInt8('d'), UInt8('a'), UInt8('t'), UInt8('a'))

function malloc_read(path::String)
    sz = filesize(path)
    ptr = Libc.malloc(sz)
    ptr == C_NULL && throw(OutOfMemoryError())
    try
        open(path, "r") do io
            unsafe_read(io, ptr, sz)
        end
    catch e
        Libc.free(ptr)
        rethrow(e)
    end
    return unsafe_wrap(Vector{UInt8}, convert(Ptr{UInt8}, ptr), sz; own=false), ptr
end

function get_layout_wav(data::AbstractVector{UInt8})
    length(data) < 44 && error("File too small to be a WAV")
    (ntuple(i -> data[i], 4) === RIFF_ID && ntuple(i -> data[i+8], 4) === WAVE_ID) || error("Not a RIFF/WAVE file")
    
    fmt_tag::UInt16, Channels::UInt16, rate::UInt32, Bits::UInt16, data_offset::Int64, total_frames::Int64 = 0, 0, 0, 0, 0, 0
    pos_offset::Int = 12
    while pos_offset + 8 < length(data)
        chunk_id::NTuple{4,UInt8} = (data[pos_offset+1], data[pos_offset+2], data[pos_offset+3], data[pos_offset+4])
        sz::UInt32 = UInt32(data[pos_offset+5]) | (UInt32(data[pos_offset+6]) << 8) | (UInt32(data[pos_offset+7]) << 16) | (UInt32(data[pos_offset+8]) << 24)
        chunk_data::Int = pos_offset + 8
        if chunk_id === FMT_ID
            fmt_tag  = UInt16(data[chunk_data+1]) | (UInt16(data[chunk_data+2]) << 8)
            Channels = (UInt16(data[chunk_data+3]) | (UInt16(data[chunk_data+4]) << 8))
            rate     = (UInt32(data[chunk_data+5]) | (UInt32(data[chunk_data+6]) << 8) | (UInt32(data[chunk_data+7]) << 16) | (UInt32(data[chunk_data+8]) << 24))
            Bits     = (UInt16(data[chunk_data+15]) | (UInt16(data[chunk_data+16]) << 8))
        elseif chunk_id === DATA_ID
            (Channels == 0 || Bits == 0) && error("Invalid WAV: DATA chunk before FMT chunk")
            total_frames = sz ÷ (Channels * (Bits ÷ 8))
            data_offset = chunk_data
            break
        end
        pos_offset = chunk_data + sz + (sz % 2)
    end
    Dtype = fmt_tag == UInt16(3) ? (Bits == 32 ? Float32 : Float64) : Bits == 16 ? Q0f15 : (Bits == 24 ? Q0f23 : Q0f31)
    return WavMetadata{Dtype, Int(Channels)}(Int(rate), data_offset, total_frames)
end

function get_layout_wav(path::String)
    header_chunk = open(io -> read(io, 1024), path, "r")
    return get_layout_wav(header_chunk)
end

@inline function is_bit_compatible(::Type{S}, ::Type{T}) where {S, T}
    (S === Q0f23 || T === Q0f23) && return false 
    sizeof(S) != sizeof(T) && return false
    S === T && return true
    return (S <: Union{FixedPoint, Integer}) && (T <: Union{FixedPoint, Integer})
end

function base_wav_parser(Input::WavMemoryInput, meta::WavMetadata{S, Channels}, ::Type{T}) where {S, Channels, T}
    data, raw_ptr, ismalloc = if Input isa Tuple
        Input[1], Input[2], true
    else
        Input, pointer(Input), false
    end
    TargetType = Sample{Channels, T}
    n_frames = meta.total_frames

    if sizeof(S) != 3 && is_bit_compatible(S, T) # FAST PATH: Zero-copy
        audio_ptr = convert(Ptr{UInt8}, raw_ptr) + (meta.data_offset)
        final_view = unsafe_wrap(Array, reinterpret(Ptr{TargetType}, audio_ptr), n_frames)
        if ismalloc
            finalizer(x -> Libc.free(raw_ptr), final_view)
        end
        return SampleArray(final_view, (meta.sample_rate,))
    else # PROCESS PATH: Copy/Convert
        dest = Vector{TargetType}(undef, n_frames)
        _process_bits!(dest, data, meta)
        if ismalloc
            Libc.free(raw_ptr)
        end
        return SampleArray(dest, (meta.sample_rate,))
    end
end

audioread(Input::WavMemoryInput, meta::WavMetadata{S, Channels}, parser_function::F = base_wav_parser) where {S, Channels, F<:Function} = parser_function(Input, meta, S)
audioread(Input::WavMemoryInput, meta::WavMetadata{S, Channels}, native_output::Bool, parser_function::F = base_wav_parser) where {S, Channels, F<:Function} = parser_function(Input, meta, native_output ? S : get_transport_type(S))
function audioread(path::String, output_param::Union{Bool, Type} = false, parser_function::F = base_wav_parser) where {F<:Function}
    raw_vec, raw_ptr = malloc_read(path)
    meta = get_layout_wav(raw_vec)
    if output_param isa Bool
        return audioread((raw_vec, raw_ptr), meta, output_param, parser_function)
    else
        return parser_function((raw_vec, raw_ptr), meta, output_param)
    end
end

@inline function _read_pcm_sample(ptr::Ptr{UInt8}, ::Type{Q0f15}, ::Type{ET}) where {ET}
    s16 = unsafe_load(reinterpret(Ptr{Int16}, ptr))
    if ET <: AbstractFloat;     return Float32(s16) * (1.0f0 / 32768.0f0)
    elseif ET <: FixedPoint;    return reinterpret(ET, s16)
    end
    return s16 % ET
end
@inline function _read_pcm_sample(ptr::Ptr{UInt8}, ::Type{Q0f23}, ::Type{ET}) where {ET}
    u24 = UInt32(unsafe_load(ptr, 1)) | (UInt32(unsafe_load(ptr, 2)) << 8) | (UInt32(unsafe_load(ptr, 3)) << 16)
    s32 = reinterpret(Int32, u24 << 8) 
    if ET <: Integer;           return s32 % ET
    elseif ET <: FixedPoint;    return reinterpret(ET, s32)
    elseif ET <: AbstractFloat; return Float32(s32) * (1.0f0 / 2147483648.0f0)
    end
    return s32
end
@inline function _read_pcm_sample(ptr::Ptr{UInt8}, ::Type{Q0f31}, ::Type{ET}) where {ET}
    s32 = unsafe_load(reinterpret(Ptr{Int32}, ptr))
    if ET <: AbstractFloat;     return Float32(s32) * (1.0f0 / 2147483648.0f0)
    elseif ET <: FixedPoint;    return reinterpret(ET, s32)
    end
    return s32 % ET
end
@inline function _read_pcm_sample(ptr::Ptr{UInt8}, ::Type{Float32}, ::Type{ET}) where {ET}
    f32 = unsafe_load(reinterpret(Ptr{Float32}, ptr))
    if ET <: Float32;           return f32
    elseif ET <: FixedPoint;    return convert(ET, f32)
    end
    return ET(f32) # Fallback
end

function _process_bits!(dest::AbstractVector{Sample{Channels, T}}, raw::Vector{UInt8}, meta::WavMetadata{S, Channels}) where {Channels, T, S}
    bytes_per_sample::Int = S === Q0f23 ? 3 : sizeof(S)
    bytes_per_frame::Int = Channels * bytes_per_sample
    GC.@preserve raw begin
        base_ptr = pointer(raw) + meta.data_offset
        @inbounds for i in eachindex(dest)
            frame_ptr = base_ptr + (i - 1) * bytes_per_frame
            dest[i] = Sample{Channels, T}(ntuple(ch -> 
                _read_pcm_sample(frame_ptr + (ch - 1) * bytes_per_sample, S, T), 
                Val(Channels))
            )
        end
    end
    return dest
end
end
