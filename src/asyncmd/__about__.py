# This file is part of asyncmd.
#
# asyncmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# asyncmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with asyncmd. If not, see <https://www.gnu.org/licenses/>.


def _get_git_version():
    import os
    import subprocess
    # Return the git revision as a string
    # adopted from numpy setup.py

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        # execute in the directory where the source files live
        # to get the correct git_rev even if we changed the working dir
        source_dir = os.path.dirname(os.path.abspath(__file__))
        output = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            env=env, cwd=source_dir).communicate()[0]
        return output
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = None
    return git_revision


__description__ = ("asyncmd is a library to write concurrent code to run and "
                   + "analyze molecular dynamics simulations using pythons "
                   + "async/await synthax"
                   )
__url__ = "https://gitea.kotspeicher.de/AIMMD/asyncmd"
__author__ = "Hendrik Jung"
__author_email__ = "hendrik.andre.jung@googlemail.com"
__license__ = "GNU General Public License v3 or later (GPLv3+)"
__copyright__ = "2022 {:s}".format(__author__)
# sort out if we have a git (commit) version
base_version = "0.1.0.rc1"
git_version = _get_git_version()
if git_version is None:
    # no git installed
    __version__ = base_version
elif git_version == "":
    # this happens if git is installed,
    # but source is not part of a repo
    __version__ = base_version
else:
    __version__ = base_version + "+" + git_version
