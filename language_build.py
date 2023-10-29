from tree_sitter import Language

Language.build_library(
    # Store the library in the `build` directory
    'build_languages/my-languages.so',
    # Include one or more languages
    [
        'languages/tree-sitter-c'
    ]
)
