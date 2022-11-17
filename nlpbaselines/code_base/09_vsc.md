# high cpu

disable builtin typescript and javascript language features
https://medium.com/good-robot/use-visual-studio-code-remote-ssh-sftp-without-crashing-your-server-a1dc2ef0936d

# exclude files

    // Place your settings in this file to overwrite default and user settings.

    {
        "settings": {
            "files.exclude": {
                "**/.git": true,         // this is a default value
                "**/.DS_Store": true,    // this is a default value

                "**/node_modules": true, // this excludes all folders
                                        // named "node_modules" from
                                        // the explore tree

                // alternative version
                "node_modules": true    // this excludes the folder
                                        // only from the root of
                                        // your workspace
            }
        }
    }
