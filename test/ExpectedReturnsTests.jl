using Test
using PortfolioOptimiser.ExpectedReturns

@testset "Efficient return value" begin
    rets = [
        -0.0006748400796242 -0.0069003561906746 -0.0062350096988893 -0.0010528295103606 0.0036162836865440 0.0055131331052059 0.0017771874692506
        -0.0136552795561427 0.0063719449271437 -0.0135387973471765 0.0020842993036400 -0.0201429193145458 0.0056784408765250 0.0002050903924777
        0.0223035263456402 -0.0125740965262354 -0.0036276784565668 0.0098423391006926 0.0068464607776117 -0.0117286082571672 -0.0132478894650054
        -0.0037694052340852 0.0047433855140722 -0.0035100901919316 -0.0073897353316849 0.0039647543861933 -0.0226398931901647 0.0079578550398675
        -0.0097545784426500 -0.0096890388933971 0.0041402500355095 0.0128447743398214 0.0040631483356639 0.0213654993368455 0.0011212612666105
        0.0097094575560662 -0.0085259796371582 -0.0061547425212878 -0.0014428011740385 0.0053837782299703 0.0017300626065962 -0.0093646255614063
        -0.0047144316267802 0.0138977829788129 0.0258638373882496 -0.0178493164506068 -0.0278156514291655 0.0003608751035137 -0.0047274631285045
        -0.0057087862005760 0.0080811932473448 -0.0078170276965343 -0.0159509506610048 0.0001072084420873 0.0063141569395188 -0.0034274512824267
        0.0052070235563621 -0.0105749812123991 0.0086771833912368 0.0004212563100071 0.0058167889271601 0.0123186402852710 -0.0093214511453497
        0.0007023413752606 0.0015784461612801 0.0124494036188893 0.0073293684785748 -0.0000545052057682 0.0002886918878566 -0.0004545563311688
        0.0048244208963245 -0.0051800785825117 -0.0038428768996661 0.0101816949438218 -0.0033787479581397 0.0069811023563090 0.0107086738257538
        -0.0139494962887654 -0.0128763133630688 0.0036707309016832 -0.0061589392328065 -0.0113414152813241 0.0121209858963399 -0.0034456095257532
        0.0049239726584480 -0.0016729159404981 -0.0126361300744444 0.0015788103092960 0.0084436841151058 0.0015615090203985 0.0116309728154348
        0.0004334332137242 0.0193219926498295 -0.0022996583271096 -0.0190491317675548 -0.0013100071392300 -0.0020094062057762 -0.0173905782894246
        0.0174254340510027 0.0119158047666807 0.0100378728475036 0.0025650747998706 -0.0016131569743071 0.0089560488161184 0.0053435102907043
        -0.0104795452758912 0.0007253579603801 0.0153702568731145 -0.0124907944932643 -0.0024575352520610 0.0090516177698353 -0.0075278787817123
        -0.0014750087032114 0.0102543355703968 0.0098636018924578 0.0076908231333085 -0.0094889576838703 -0.0037695120552527 -0.0098535207723071
        -0.0026830204066995 -0.0022082607311000 -0.0056567761525415 0.0006415721056171 -0.0014209688964841 -0.0022181584503209 -0.0082123679784803
        0.0087061361381246 0.0022512347199988 0.0061362975667603 -0.0041455835478169 -0.0071994076807952 0.0086123901120495 -0.0047472407990139
        -0.0082829057326971 -0.0091231017006632 0.0019692806048999 -0.0079338678139561 0.0100424896047813 0.0033083268404122 0.0007857151456365
    ]
    capm = ret_model(CAPMRet(), rets)
    capmtest = [
        -0.122013075152358
        0.031492780547182
        -0.055411458233836
        -0.116514174592914
        -0.057322040286728
        -0.058764132119647
        -0.054896894406269
    ]
    @test capm ≈ capmtest
    mret = ret_model(MRet(), rets)
    mrettest = [
        -0.022403739743102
        -0.013028170439639
        0.494428689026683
        -0.389532486922771
        -0.387068146087651
        1.153500608621242
        -0.486114942675440
    ]
    @test mret ≈ mrettest
    eret = ret_model(EMRet(), rets, span = 500)
    erettest = [
        -0.011759699535282
        0.007429169377570
        0.537130952670764
        -0.388512066342667
        -0.378983081078194
        1.191648084065445
        -0.485051822075664
    ]
    @test eret ≈ erettest
end