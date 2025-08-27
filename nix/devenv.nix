{
  perSystem = { pkgs, ...}:
    {
      devenv.shells.default = {
        cachix.enable = true;

        # https://devenv.sh/basics/
        env.GREET = "devenv";
        enterShell = ''
          hello
          git --version
        '';

        # https://devenv.sh/tasks/
        # tasks = {
        #   "myproj:setup".exec = "mytool build";
        #   "devenv:enterShell".after = [ "myproj:setup" ];
        # };

        # https://devenv.sh/tests/
        enterTest = ''
          echo "Running tests"
          git --version
        '';

        languages.python = {
          enable = true;
          uv = {
            enable = true;
            sync.enable = true;
          };
          package = pkgs.python312;
          # version = "3.12";
        };

        # https://devenv.sh/git-hooks/
        git-hooks.hooks = {
          shellcheck.enable = true;
          markdownlint.enable = true;
          
          deadnix.enable = true;
          statix.enable = true;
          # nixfmt.enable = true;
          nil.enable = true;

          uv-check.enable = true;
          black.enable = true;
          flake8.enable = true;
          pyright.enable = true;
        };

        # See full reference at https://devenv.sh/reference/options/
      };
    };
}