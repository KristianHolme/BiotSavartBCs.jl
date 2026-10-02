using KernelAbstractions
using KernelAbstractions: get_backend,@kernel,@index,@Const
KernelAbstractions.get_backend(nt::NTuple) = get_backend(first(nt))
