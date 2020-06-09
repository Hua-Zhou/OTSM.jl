using Documenter, OTSM

makedocs(
    format = Documenter.HTML(),
    sitename = "OTSM.jl",
    authors = "Hua Zhou, Joong-Ho Won",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/Hua-Zhou/OTSM.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
