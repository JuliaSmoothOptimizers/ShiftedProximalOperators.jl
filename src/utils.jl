# non-allocating reshape
# see https://github.com/JuliaLang/julia/issues/36313
reshape_array(a, dims) = invoke(Base._reshape, Tuple{AbstractArray, typeof(dims)}, a, dims)
