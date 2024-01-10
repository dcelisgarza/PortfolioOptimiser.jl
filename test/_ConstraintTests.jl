using Test, PortfolioOptimiser, DataFrames, OrderedCollections, Clarabel, CSV,
      TimeSeries

@testset "Asset constraints" begin
    asset_classes = Dict("Assets" => ["FB", "GOOGL", "NTFX", "BAC", "WFC",
                                      "TLT", "SHV", "FCN", "TKO", "ZOO", "ZVO",
                                      "ZX", "ZZA", "ZZB", "ZZC"],
                         "Class 1" => ["Equity", "Equity", "Equity", "Equity",
                                       "Equity", "Fixed Income", "Fixed Income",
                                       "Equity", "Equity", "Equity",
                                       "Fixed Income", "Fixed Income", "Equity",
                                       "Fixed Income", "Equity"],
                         "Class 2" => ["Technology", "Technology", "Technology",
                                       "Financial", "Financial", "Treasury",
                                       "Treasury", "Financial", "Entertainment",
                                       "Treasury", "Financial", "Financial",
                                       "Entertainment", "Technology",
                                       "Treasury"])

    constraints = Dict("Enabled" => [true, true, true, true, true, true, true,
                                     true, true, true, true, true, true, true,
                                     true],
                       "Type" => ["Classes", "All Classes", "Assets", "Assets",
                                  "Classes", "All Assets",
                                  "Each Asset in Class", "Assets", "All Assets",
                                  "All Assets", "Classes", "All Classes",
                                  "All Classes", "Each Asset in Class",
                                  "Each Asset in Class"],
                       "Set" => ["Class 1", "Class 1", "", "", "Class 2", "",
                                 "Class 1", "Class 1", "Class 2", "", "Class 1",
                                 "Class 2", "Class 2", "Class 2", "Class 1"],
                       "Position" => ["Equity", "Fixed Income", "BAC", "WFC",
                                      "Financial", "", "Equity", "FCN", "TKO",
                                      "ZOO", "Fixed Income", "Treasury",
                                      "Entertainment", "Treasury", "Equity"],
                       "Sign" => ["<=", "<=", "<=", "<=", ">=", ">=", ">=",
                                  "<=", ">=", "<=", ">=", "<=", ">=", "<=",
                                  ">="],
                       "Weight" => [0.6, 0.5, 0.1, "", "", 0.02, "", "", "", "",
                                    "", "", "", 0.27, ""],
                       "Type Relative" => ["", "", "", "Assets", "Classes", "",
                                           "Assets", "Classes", "Assets",
                                           "Classes", "Assets", "Assets",
                                           "Classes", "", "Classes"],
                       "Relative Set" => ["", "", "", "", "Class 1", "", "",
                                          "Class 1", "", "Class 2", "",
                                          "Class 2", "Class 2", "", "Class 2"],
                       "Relative" => ["", "", "", "FB", "Fixed Income", "",
                                      "TLT", "Equity", "NTFX", "Financial",
                                      "WFC", "ZOO", "Entertainment", "",
                                      "Entertainment"],
                       "Factor" => ["", "", "", 1.2, 0.5, "", 0.4, 0.7, 0.21,
                                    0.11, 0.13, -0.17, 0.23, "", -0.31])

    constraints = DataFrame(constraints)
    asset_classes = DataFrame(asset_classes)
    sort!(asset_classes, "Assets")

    A, B = asset_constraints(constraints, asset_classes)

    At = transpose(hcat([[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0,
                          -1.0, 0.0, 0.0, -1.0, 0.0, -1.0],
                         [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0,
                          -1.0, 0.0, 0.0, -1.0, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0,
                          -1.0, -1.0, 0.0, -1.0, 0.0],
                         [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [-0.0, 1.2, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0,
                          -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                         [1.0, 0.0, 1.0, 0.0, 0.0, -0.5, 0.0, -0.5, 1.0, 0.0,
                          0.5, 0.5, 0.0, -0.5, 0.0],
                         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                          0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 1.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.7, 0.7, -0.3, 0.7, 0.7, -0.0, 0.7, -0.0, 0.7, 0.7,
                          -0.0, -0.0, 0.7, -0.0, 0.7],
                         [1.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.79, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 1.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 1.0],
                         [-0.89, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -1.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, -0.89, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -1.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -1.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -1.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -1.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -1.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, -0.89,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -1.0, 0.11, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, -0.89, 0.11, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, -0.89, -0.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -1.0, -0.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -1.0, -0.0],
                         [0.11, -0.0, 0.11, -0.0, -0.0, -0.0, -0.0, -0.0, 0.11,
                          -0.0, 0.11, 0.11, -0.0, -0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -0.13, 0.0,
                          1.0, 1.0, 0.0, 1.0, 0.0],
                         [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0,
                          -0.17, -0.0, -0.0, -1.0, -0.0, -0.0],
                         [-1.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0,
                          -0.17, -1.0, -1.0, -0.0, -0.0, -0.0],
                         [-0.0, -1.0, -0.0, -1.0, -1.0, -0.0, -0.0, -0.0, -0.0,
                          -0.17, -0.0, -0.0, -0.0, -1.0, -0.0],
                         [-0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -1.0, -0.0,
                          -1.17, -0.0, -0.0, -0.0, -0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.77, 0.0, 0.0],
                         [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.23, 0.0, 1.0, 0.0,
                          1.0, 1.0, -0.23, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -0.23, 0.0, 0.0, 0.0,
                          0.0, 0.0, -0.23, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.23, 1.0, 0.0, 1.0,
                          0.0, 0.0, -0.23, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 1.0,
                          0.0, 0.0, 0.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 1.31, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.31, 0.0, 1.0]]...))

    Bt = vcat([[-0.6], [-0.5], [-0.5], [-0.1], [0.0], [0.0], [0.02], [0.02],
               [0.02], [0.02], [0.02], [0.02], [0.02], [0.02], [0.02], [0.02],
               [0.02], [0.02], [0.02], [0.02], [0.02], [0.0], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
               [0.0], [0.0], [-0.27], [-0.27], [-0.27], [-0.27], [0.0], [0.0],
               [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]...)

    @test isapprox(At, A)
    @test isapprox(Bt, B)
end

@testset "Factor constraints" begin
    loadings = Dict("const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003,
                                -0.0003],
                    "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000,
                               0.0000],
                    "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000,
                               0.0000],
                    "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857,
                               0.0000],
                    "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055,
                               0.5781, 0.0000],
                    "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054,
                               -0.4844, 0.9435])

    loadings = DataFrame(loadings)

    constraints = Dict("Enabled" => [true, true, true, true],
                       "Factor" => ["MTUM", "USMV", "VLUE", "const"],
                       "Sign" => ["<=", "<=", ">=", ">="],
                       "Value" => [0.9, -1.2, 0.3, -0.1],
                       "Relative Factor" => ["USMV", "", "", "SIZE"])

    constraints = DataFrame(constraints)

    C, D = factor_constraints(constraints, loadings)

    Ct = transpose(hcat([[-9.7540e-01, -2.6500e+00, -1.8871e+00, -3.4403e+00,
                          5.5000e-03, 5.7810e-01, -0.0000e+00],
                         [7.8380e-01, 1.6439e+00, 1.0176e+00, 1.4407e+00,
                          -5.5000e-03, -5.7810e-01, -0.0000e+00],
                         [1.4772e+00, -7.5900e-01, -4.0900e-01, 0.0000e+00,
                          -5.4000e-03, -4.8440e-01, 9.4350e-01],
                         [4.0000e-04, 2.0000e-04, 0.0000e+00, -4.7110e-01,
                          1.0000e-04, 1.8600e-01, -3.0000e-04]]...))

    Dt = vcat([[-0.9]
               [1.2]
               [0.3]
               [-0.1]]...)

    @test isapprox(Ct, C)
    @test Dt == D
end

@testset "Views constraints" begin
    asset_classes = Dict("Assets" => ["FB", "GOOGL", "NTFX", "BAC", "WFC",
                                      "TLT", "SHV"],
                         "Class 1" => ["Equity", "Equity", "Equity", "Equity",
                                       "Equity", "Fixed Income", "Fixed Income"],
                         "Class 2" => ["Technology", "Technology", "Technology",
                                       "Financial", "Financial", "Treasury",
                                       "Treasury"])

    asset_classes = DataFrame(asset_classes)
    sort!(asset_classes, "Assets")

    views = Dict("Enabled" => [true, true, true, true, true],
                 "Type" => ["Assets", "Classes", "Classes", "Assets", "Classes"],
                 "Set" => ["", "Class 2", "Class 1", "", "Class 1"],
                 "Position" => ["WFC", "Financial", "Equity", "FB",
                                "Fixed Income"],
                 "Sign" => ["<=", ">=", ">=", ">=", "<="],
                 "Return" => [0.3, 0.1, 0.05, 0.03, 0.017],
                 "Type Relative" => ["Assets", "Classes", "Assets", "", ""],
                 "Relative Set" => ["", "Class 1", "", "", ""],
                 "Relative" => ["FB", "Fixed Income", "TLT", "", ""])

    views = DataFrame(views)
    P, Q = asset_views(views, asset_classes)

    Pt = transpose(hcat([[-0.0, -1.0, -0.0, -0.0, -0.0, -0.0, 1.0],
                         [0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5],
                         [0.2, 0.2, 0.2, 0.2, 0.0, -1.0, 0.2],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0]]...))
    Qt = vcat([[0.3], [0.1], [0.05], [0.03], [0.017]]...)
    @test isapprox(Pt, P)
    @test isapprox(Qt, Q)
end

@testset "Factor views" begin
    loadings = Dict("const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003,
                                -0.0003],
                    "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000,
                               0.0000],
                    "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000,
                               0.0000],
                    "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857,
                               0.0000],
                    "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055,
                               0.5781, 0.0000],
                    "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054,
                               -0.4844, 0.9435])

    loadings = DataFrame(loadings)

    views = Dict("Enabled" => [true, true, true],
                 "Factor" => ["MTUM", "USMV", "VLUE"],
                 "Sign" => ["<=", "<=", ">="], "Value" => [0.9, -1.2, 0.3],
                 "Relative Factor" => ["USMV", "", ""])

    views = DataFrame(views)

    P, Q = factor_views(views, loadings)

    Pt = transpose(hcat([[-1, 0, 0, 1, 0], [0, 0, 0, -1, 0],
                         [0, 0, 0, 0, 1]]...))
    Qt = vcat([[-0.9], [1.2], [0.3]]...)
    @test isapprox(Pt, P)
    @test isapprox(Qt, Q)

    loadings = Dict("MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000,
                               0.0000],
                    "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000,
                               0.0000],
                    "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857,
                               0.0000],
                    "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055,
                               0.5781, 0.0000],
                    "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054,
                               -0.4844, 0.9435])
    loadings = DataFrame(loadings)
    P, Q = factor_views(views, loadings)
    Pt = transpose(hcat([[-1, 0, 0, 1, 0], [0, 0, 0, -1, 0],
                         [0, 0, 0, 0, 1]]...))
    Qt = vcat([[-0.9], [1.2], [0.3]]...)
    @test isapprox(Pt, P)
    @test isapprox(Qt, Q)
end

@testset "HRP constraints" begin
    asset_classes = Dict("Assets" => ["FB", "GOOGL", "NTFX", "BAC", "WFC",
                                      "TLT", "SHV"],
                         "Class 1" => ["Equity", "Equity", "Equity", "Equity",
                                       "Equity", "Fixed Income", "Fixed Income"],
                         "Class 2" => ["Technology", "Technology", "Technology",
                                       "Financial", "Financial", "Treasury",
                                       "Treasury"])

    asset_classes = DataFrame(asset_classes)
    sort!(asset_classes, "Assets")

    constraints = Dict("Enabled" => [true, true, true, true, true, true],
                       "Type" => ["Assets", "Assets", "All Assets",
                                  "All Assets", "Each Asset in Class",
                                  "Each Asset in Class"],
                       "Set" => ["", "", "", "", "Class 1", "Class 2"],
                       "Position" => ["BAC", "FB", "", "", "Fixed Income",
                                      "Financial"],
                       "Sign" => [">=", "<=", "<=", ">=", "<=", "<="],
                       "Weight" => [0.02, 0.085, 0.09, 0.01, 0.07, 0.06])
    constraints = DataFrame(constraints)

    w_min, w_max = hrp_constraints(constraints, asset_classes)

    w_mint = [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    w_maxt = [0.06, 0.085, 0.09, 0.09, 0.07, 0.07, 0.06]
    @test isapprox(w_mint, w_min)
    @test isapprox(w_maxt, w_max)
end

@testset "RP constraints" begin
    asset_classes = Dict("Assets" => ["FB", "GOOGL", "NTFX", "BAC", "WFC",
                                      "TLT", "SHV"],
                         "Class 1" => ["Equity", "Equity", "Equity", "Equity",
                                       "Equity", "Fixed Income", "Fixed Income"],
                         "Class 2" => ["Technology", "Technology", "Technology",
                                       "Financial", "Financial", "Treasury",
                                       "Treasury"])
    asset_classes = DataFrame(asset_classes)
    asset_classes = sort!(asset_classes, "Assets")

    w1 = rp_constraints(asset_classes, :Classes, "Class 1")
    w2 = rp_constraints(asset_classes, :Classes, 2)
    w3 = rp_constraints(asset_classes, :Classes, Symbol("Class 1"))
    w4 = rp_constraints(asset_classes, :Classes, "Class 2")
    w5 = rp_constraints(asset_classes, :Classes, 3)
    w6 = rp_constraints(asset_classes, :Classes, Symbol("Class 2"))
    w7 = rp_constraints(asset_classes, :Assets)

    wt1 = vcat([[0.1], [0.1], [0.1], [0.1], [0.25], [0.25], [0.1]]...)
    wt2 = vcat([[0.16666666666666666], [0.1111111111111111],
                [0.1111111111111111], [0.1111111111111111],
                [0.16666666666666666], [0.16666666666666666],
                [0.16666666666666666]]...)
    @test isapprox(wt1, w1)
    @test isapprox(wt1, w2)
    @test isapprox(wt1, w3)
    @test isapprox(wt2, w4)
    @test isapprox(wt2, w5)
    @test isapprox(wt2, w6)
    @test all(isapprox.(w7, 1 / 7))
    @test_throws ArgumentError rp_constraints(asset_classes, :Classes, "Wak")
end

@testset "A and B inequalities" begin
    A = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
    Y = percentchange(A)
    returns = dropmissing!(DataFrame(Y))

    portfolio = Portfolio(; returns = returns,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    asset_classes = Dict("Assets" => names(returns[!, 2:end]),
                         "Industry" => ["Consumer Discretionary",
                                        "Consumer Discretionary",
                                        "Consumer Staples", "Consumer Staples",
                                        "Energy", "Financials", "Financials",
                                        "Health Care", "Health Care",
                                        "Industrials", "Industrials",
                                        "Health Care", "Industrials",
                                        "Information Technology", "Materials",
                                        "Telecommunications Services",
                                        "Utilities", "Utilities",
                                        "Telecommunications Services",
                                        "Financials"])
    asset_classes = DataFrame(asset_classes)
    sort!(asset_classes, "Assets")
    constraints = Dict("Enabled" => [true, true, true, true, true],
                       "Type" => ["All Assets", "Classes", "Classes", "Classes",
                                  "Classes"],
                       "Set" => ["", "Industry", "Industry", "Industry",
                                 "Industry"],
                       "Position" => ["", "Financials", "Utilities",
                                      "Industrials", "Consumer Discretionary"],
                       "Sign" => ["<=", "<=", "<=", "<=", "<="],
                       "Weight" => [0.10, 0.2, 0.2, 0.2, 0.2],
                       "Type Relative" => ["", "", "", "", ""],
                       "Relative Set" => ["", "", "", "", ""],
                       "Relative" => ["", "", "", "", ""],
                       "Factor" => ["", "", "", "", ""])
    constraints = DataFrame(constraints)
    w1 = opt_port!(portfolio)
    w1t = [6.404242909815355e-9, 2.139687463536551e-8, 2.773303237720693e-8,
           1.5328754075787494e-8, 0.4744710120183094, 1.903064320608223e-9,
           0.05589665787782722, 2.7521145279052637e-8, 1.1762408653019186e-8,
           7.397542568043182e-9, 1.8920129479125674e-8, 1.4436635231675774e-9,
           1.0509898162704996e-9, 3.6757867384609326e-9, 1.0210021773260428e-9,
           0.13875927538515884, 0.2221373459785268, 2.088612011559538e-8,
           0.10873552381602443, 1.847939655895259e-8]
    @test isapprox(w1.weights, w1t, rtol = 7e-5)

    w2 = opt_port!(portfolio; obj = :Min_Risk)
    w2t = [0.00790101350930099, 0.0306909139358808, 0.010506322670541511,
           0.027487810990731512, 0.012278574123481116, 0.033410647094723245,
           5.992768702616961e-10, 0.1398486652999066, 1.0318412667826644e-9,
           1.1506529577820253e-8, 0.2878239722245232, 6.544625217911222e-10,
           4.93689273689815e-10, 0.1252846767367913, 2.4717046303259267e-9,
           0.015084330918903146, 1.814161154983086e-8, 0.19312632144038927,
           1.2981261894646573e-9, 0.11655671485758536]
    @test isapprox(w2.weights, w2t, rtol = 9e-5)

    A, B = asset_constraints(constraints, asset_classes)
    portfolio.a_mtx_ineq = A
    portfolio.b_vec_ineq = B
    w3 = opt_port!(portfolio)
    w3t = [0.005337674597465762, 0.0999999328085889, 0.09999998551049004,
           0.059054659420022665, 0.09999999785054277, 1.8783625308022804e-9,
           0.06600399341467694, 0.04833717446345979, 1.3653878243927298e-8,
           1.1794690581037013e-8, 0.05796280977060682, 1.4608612365759018e-9,
           9.064398757224001e-10, 4.450158510374558e-9, 1.0162133156406168e-9,
           0.09999998413955972, 0.09999998799261628, 0.06330380274630418,
           0.09999998821855735, 0.09999997390650456]
    @test isapprox(w3.weights, w3t, rtol = 1e-4)
    w4 = opt_port!(portfolio; obj = :Min_Risk)
    w4t = [0.061917908789090266, 0.08125925739913735, 0.0209663505365103,
           0.026541298795123135, 0.01346199216116964, 0.09999999839649047,
           6.762323459622066e-10, 0.09999999953536425, 1.4495721257709719e-9,
           0.05294375358043678, 0.09999999963075093, 8.303950344629004e-10,
           1.0162236692008542e-9, 0.09999999929260821, 0.010582248161957103,
           0.03920313937613068, 0.09312402892645888, 0.09999999947467735,
           2.313484329129996e-8, 0.09999999883682797]
    @test isapprox(w4.weights, w4t, rtol = 1e-4)
end
