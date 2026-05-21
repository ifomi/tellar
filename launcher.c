// Tellar — self-contained native launcher for the embedded Python runtime.
//
// This binary lives at Tellar.app/Contents/MacOS/tellar and is the
// CFBundleExecutable. It resolves its own location at runtime, points the
// embedded CPython at the bundled runtime, and runs `tellar.app` via runpy
// in the same process. There is no shell, no exec(), and no hardcoded path
// to anything outside the bundle (except $HOME for log placement, which is
// per-user state and resolved at runtime).
//
// Layout assumed:
//   Tellar.app/
//     Contents/
//       MacOS/tellar                       <- this binary
//       Resources/
//         python/                          <- python-build-standalone tree
//           bin/python3.12
//           lib/libpython3.12.dylib        <- linked here via @rpath
//           lib/python3.12/                <- stdlib
//         app/
//           tellar/                        <- our package
//           site-packages/                 <- pip-installed deps

#include <Python.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return 0;
    return mkdir(path, 0755);
}

static void redirect_stdio_to_log(const char *bundle_id_for_logname) {
    const char *home = getenv("HOME");
    if (!home || !*home) return;

    char logdir[4096];
    snprintf(logdir, sizeof(logdir), "%s/Library/Logs/Tellar", home);
    ensure_dir(logdir);

    char logpath[4096];
    snprintf(logpath, sizeof(logpath), "%s/%s.log", logdir, bundle_id_for_logname);
    int fd = open(logpath, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
    }
}

int main(int argc, char *argv[]) {
    // Resolve the absolute path to this executable.
    char exe[4096];
    uint32_t exe_size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &exe_size) != 0) {
        fprintf(stderr, "tellar: _NSGetExecutablePath buffer too small\n");
        return 1;
    }

    char resolved[4096];
    if (!realpath(exe, resolved)) {
        // Fall back to the unresolved path; symlink edge cases shouldn't be fatal.
        strncpy(resolved, exe, sizeof(resolved) - 1);
        resolved[sizeof(resolved) - 1] = '\0';
    }

    // resolved = .../Tellar.app/Contents/MacOS/tellar
    // Walk up two levels to land on Contents/.
    char contents_buf[4096];
    strncpy(contents_buf, resolved, sizeof(contents_buf) - 1);
    contents_buf[sizeof(contents_buf) - 1] = '\0';
    char *contents = dirname(contents_buf);   // .../MacOS
    contents = dirname(contents);             // .../Contents

    char python_home[4096];
    char app_dir[4096];
    char site_packages[4096];
    char pythonpath[8192];

    snprintf(python_home, sizeof(python_home), "%s/Resources/python", contents);
    snprintf(app_dir, sizeof(app_dir), "%s/Resources/app", contents);
    snprintf(site_packages, sizeof(site_packages), "%s/site-packages", app_dir);
    snprintf(pythonpath, sizeof(pythonpath), "%s:%s", app_dir, site_packages);

    // Python interpreter passthrough.
    //
    // The standard library's multiprocessing module spawns workers by running
    // sys.executable with arguments like ["-c", "from multiprocessing... main(N)"].
    // sys.executable here is *this* binary, not a stock Python interpreter.
    // If we proceed to run runpy.run_module('tellar.app') in that child, we
    // launch a second full Tellar instance — second menu-bar icon, second
    // event tap, the works. Detect Python-style invocation by the leading "-"
    // argument and exec the real interpreter inside the bundle instead. The
    // child then behaves like a normal Python subprocess.
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] != '\0') {
        char real_python[4096];
        snprintf(real_python, sizeof(real_python), "%s/bin/python3.12", python_home);
        // Ensure the child sees the same PYTHONHOME/PYTHONPATH we'd use.
        setenv("PYTHONHOME", python_home, 1);
        setenv("PYTHONPATH", pythonpath, 1);
        execv(real_python, argv);
        // Only reached if execv failed.
        fprintf(stderr, "tellar: execv(%s) failed: %s\n", real_python, strerror(errno));
        return 1;
    }

    // Open log early so any later messages from us and from Python land there.
    redirect_stdio_to_log("launcher");

    fprintf(stderr, "=== Tellar launcher start ===\n");
    fprintf(stderr, "exe=%s\n", resolved);
    fprintf(stderr, "PYTHONHOME=%s\n", python_home);
    fprintf(stderr, "PYTHONPATH=%s\n", pythonpath);
    fflush(stderr);

    // Set environment for embedded Python. setenv with overwrite=1 because we
    // want a clean, predictable environment regardless of what launched us.
    setenv("PYTHONHOME", python_home, 1);
    setenv("PYTHONPATH", pythonpath, 1);
    setenv("PYTHONIOENCODING", "utf-8", 1);
    // overwrite=0: only set if user hasn't configured locale themselves.
    setenv("LANG", "en_US.UTF-8", 0);
    setenv("LC_ALL", "en_US.UTF-8", 0);
    // Marker for the Python side to know it's running inside the bundle.
    setenv("TELLAR_BUNDLED", "1", 1);

    // Initialize embedded interpreter. Py_SetProgramName affects sys.executable
    // and the path computation Python does internally.
    wchar_t *program = Py_DecodeLocale(resolved, NULL);
    if (program) {
        Py_SetProgramName(program);
    }

    Py_Initialize();
    fprintf(stderr, "Python initialized\n");
    fflush(stderr);

    // Hand off to the Python entry point. runpy.run_module gives us the same
    // semantics as `python -m tellar.app`, so __name__ == "__main__" works.
    int ret = PyRun_SimpleString(
        "import sys, runpy\n"
        "sys.argv = ['tellar']\n"
        "runpy.run_module('tellar.app', run_name='__main__')\n"
    );

    if (ret != 0) {
        fprintf(stderr, "tellar: PyRun_SimpleString returned %d\n", ret);
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    }

    Py_Finalize();
    if (program) PyMem_RawFree(program);
    return ret;
}
