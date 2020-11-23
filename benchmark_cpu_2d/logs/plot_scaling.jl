using DelimitedFiles, PyPlot, PyCall
np = pyimport("numpy")

devito_vals = readdlm("devito-conv-split.txt")[1:2:end, [1,2,3,5]]
torch_vals = readdlm("torch-conv-split.txt")[1:2:end, [1,2,3,5]]

nch = [2,4,8,16]
ks = [3,5,7,9,11]
Ns = collect(5:14)

devito_t = reshape(devito_vals[:, end], 10, 5, 4)
torch_t = reshape(torch_vals[:, end], 10, 5, 4)

# Channel scaling
h = nch
h2 = [1, 4, 16, 64]

for k=1:5
    fig = figure(figsize=(12,6))
    for N=1:length(Ns)
        subplot(2,5,N)
        loglog(h, torch_t[N, k, :]./torch_t[N, k, 1], "^-", linewidth=2, basex=2, basey=2)
        loglog(h, devito_t[N, k, :]./devito_t[N, k, 1], "o-", linewidth=2,  basex=2, basey=2)
        loglog(h, h/h[1], ".-", linewidth=2,  basex=2, basey=2)
        loglog(h, h2/h2[1], "*-", linewidth=2,  basex=2, basey=2)
        grid(true, which="both", ls="-", alpha=.2)
        title("N=2^$(Ns[N]), k=$(ks[k])")
        (N==1 || N==6) && ylabel("Runtime (s)")
        N>5 && xlabel("Number of channels")
    end
    line_labels = ["pytroch","devito", "linear","quadratic", "cubic"]
    figlegend(line_labels, loc = "upper center", borderaxespad=0.1, ncol=5, labelspacing=0.)
    savefig("channel_sclaing_k$(ks[k])", bbox_inches="tight")
end
