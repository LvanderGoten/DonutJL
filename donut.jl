@everywhere begin
using Distributed
using ArgParse
using ProgressMeter
using Statistics
using Images
using ImageTransformations
using LinearAlgebra
using Printf
end


@everywhere function torus(R::Float64, r::Float64, δθ::Float64, δϕ::Float64)::Array{Float64, 3}
    θ = range(0.0, step=δθ, stop=2 * π) # [Nθ]
    ϕ = range(0.0, step=δϕ, stop=2 * π) # [Nϕ]

    h = R .+ r * cos.(θ)
    x = h' .* cos.(ϕ)       # [Nϕ, Nθ]
    y = h' .* sin.(ϕ)       # [Nϕ, Nθ]
    z = (r * sin.(θ))' .* (oneunit(δϕ) .+ zero(ϕ))   # [Nϕ, Nθ]

    return cat(x, y, z, dims=3)     # [Nϕ, Nθ, 3]
end


@everywhere function torus_normals(R::Float64, r::Float64, δθ::Float64, δϕ::Float64)::Array{Float64, 3}
    θ = range(0.0, step=δθ, stop=2 * π) # [Nθ]
    ϕ = range(0.0, step=δϕ, stop=2 * π) # [Nϕ]

    x   = cos.(θ)' .* cos.(ϕ)     # [Nϕ, Nθ]
    y   = cos.(θ)' .* sin.(ϕ)     # [Nϕ, Nθ]
    z   = sin.(θ)' .* (oneunit(δϕ) .+ zero(ϕ))  # [Nϕ, Nθ]

    return cat(x, y, z, dims=3)     # [Nϕ, Nθ, 3]
end

@everywhere function apply_rotation(pts::Array{Float64, 3}, δβ::Float64)::Array{Float64, 3}
    S = sum(pts, dims=3)
    return (S .+ (3 * pts .- S) * cos(δβ) + sqrt(3) * (circshift(pts, (0, 0, 1)) - circshift(pts, (0, 0, -1))) * sin(δβ))/3
end

@everywhere function calculate_luminance(normals::Array{Float64, 3},
        light_direction::Array{Float64})::Array{Float64, 2}
    return dropdims(sum(normals .* reshape(light_direction, 1, 1, 3), dims=3), dims=3)
end

@everywhere function save_array(arr::Array{Float64, 3}, fname::String)
    flatArr = string.(reshape(arr, :, 3))
    open(fname, "w") do fout
        for row in eachrow(flatArr)
            println(fout, join(row, ","))
        end
    end
end

@everywhere function perspective_projection(pts::Array{Float64, 3},
                                distance_to_origin::Float64,
                                distance_to_screen::Float64)::Array{Float64, 3}
    x_3d = pts[:, :, 1]     # [Nϕ, Nθ]
    y_3d = pts[:, :, 2]     # [Nϕ, Nθ]
    z_3d = pts[:, :, 3]     # [Nϕ, Nθ]

    x_2d = (distance_to_screen .* x_3d) ./ (distance_to_origin .+ z_3d)  # [Nϕ, Nθ]
    y_2d = (distance_to_screen .* y_3d) ./ (distance_to_origin .+ z_3d)  # [Nϕ, Nθ]
    z_2d = z_3d

    return cat(x_2d, y_2d, z_2d, dims=3)    # [Nϕ, Nθ, 3]
end

@everywhere function z_buffer(pts::Array{Float64, 3},
                  luminance::Array{Float64, 2},
                  distance_to_origin::Float64,
                  distance_to_screen::Float64,
                  screen_width::Int64,
                  screen_height::Int64,
                  field_of_view::Float64,
                  supersampling_factor::Int64)::Array{Float64, 2}
    working_screen_width = screen_width << supersampling_factor
    working_screen_height = screen_height << supersampling_factor
    
    ptsFlat = reshape(pts, :, 3)
    L = reshape(luminance, :)  # [Nϕ ⋅ Nθ]
    x = ptsFlat[:, 1]          # [Nϕ ⋅ Nθ]
    y = ptsFlat[:, 2]          # [Nϕ ⋅ Nθ]
    z = ptsFlat[:, 3]          # [Nϕ ⋅ Nθ]

    l = tan(field_of_view/2) * distance_to_screen
    ii = convert(Array{Int64}, round.((x .+ l)/(2 * l) * (working_screen_width - 1) .+ 1))
    jj = convert(Array{Int64}, round.((y .+ l)/(2 * l) * (working_screen_height - 1) .+ 1))
    is_valid = (1 .<= ii .<= working_screen_width) .& (1 .<= jj .<= working_screen_height)

    img = -ones(Float64, (working_screen_width, working_screen_height))
    z_buffer = fill(typemax(Float64), (working_screen_width, working_screen_height))
    @inbounds for k in 1:length(L)
        if !is_valid[k]
            continue
        end

        i = ii[k]
        j = jj[k]
        if z[k] < z_buffer[i, j]
            img[i, j] = L[k]
            z_buffer[i, j] = z[k]
        end

    end

    img = (img .+ 1.)/2.
    img = reshape(img, (2^supersampling_factor, screen_width,
                        2^supersampling_factor, screen_height))
    img = dropdims(mean(img, dims=(1, 3)), dims=(1, 3))

    return img
end

function print_to_screen(img::Array{Float32, 2})
    # TODO
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--r"
        default = 2.
        arg_type = Float64
        "--R"
        default = 5.
        arg_type = Float64
        "--delta_theta"
        default = float(π)/512
        arg_type = Float64
        "--delta_phi"
        default = float(π)/512
        arg_type = Float64
        "--delta_beta"
        default = float(π)/128
        arg_type = Float64
        "--distance_to_screen"
        default = 8.
        arg_type = Float64
        "--distance_to_origin"
        default = 10.
        arg_type = Float64
        "--screen_width"
        default = 512
        arg_type = Int64
        "--screen_height"
        default = 512
        arg_type = Int64
        "--field_of_view"
        default = deg2rad(120.)
        arg_type = Float64
        "--light_direction"
        default = [0., -1., 1.]
        arg_type = Float64
        nargs = '+'
        "--supersampling_factor"
        default = 3
        arg_type = Int64
        "--renderings_dir"
        default = "donut_temp/renderings"
        arg_type = String
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    r                    = args["r"]
    R                    = args["R"]
    δθ                   = args["delta_theta"]
    δϕ                   = args["delta_phi"]
    δβ                   = args["delta_beta"]
    distance_to_screen   = args["distance_to_screen"]
    distance_to_origin   = args["distance_to_origin"]
    screen_width         = args["screen_width"]
    screen_height        = args["screen_height"]
    field_of_view        = args["field_of_view"]
    light_direction      = normalize(args["light_direction"])
    supersampling_factor = args["supersampling_factor"]
    renderings_dir       = args["renderings_dir"]

    rm(renderings_dir, force=true, recursive=true)
    mkpath(renderings_dir)

    pts                      = torus(R, r, δθ, δϕ)           # [Nϕ, Nθ, 3]
    pts_normals              = torus_normals(R, r, δθ, δϕ)   # [Nϕ, Nθ, 3]

    β = range(0.0, step=δβ, stop=2 * π)
    @showprogress 1 "Computed frames:" pmap(1:length(β)) do i

        pts_rot         = apply_rotation(pts, β[i])               # [Nϕ, Nθ, 3]
        pts_normals_rot = apply_rotation(pts_normals, β[i])       # [Nϕ, Nθ, 3]
        luminance       = calculate_luminance(pts_normals_rot, light_direction)  # [Nϕ, Nθ]
        pts_rot_proj    = perspective_projection(pts_rot,
                                                 distance_to_origin,
                                                 distance_to_screen)
        img = z_buffer(pts_rot_proj,
                       luminance,
                       distance_to_origin,
                       distance_to_screen,
                       screen_width,
                       screen_height,
                       field_of_view,
                       supersampling_factor)

        img_id = @sprintf("%05d", i)
        save(joinpath(renderings_dir, img_id * ".png"), colorview(Gray, img))
    end

    run(Cmd(`ffmpeg -y -loglevel warning -framerate 30 -i %05d.png ../../donut.webm`, dir=renderings_dir))

end

main()
