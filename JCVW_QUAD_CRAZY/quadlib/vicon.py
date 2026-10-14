"""Vicon client. Ported from ph2_sysid_flying.py in the integration project.

Runs in its own thread at the Vicon frame rate. The EKF feed comes first so the
control-critical path is never blocked by bookkeeping, and the latest pose is
published under a lock for the control loop to sample non-blocking.

The control loop must NOT call waitForNextFrame itself: it runs on its own
deadline and takes whatever the most recent pose is.
"""

import threading
import time

import motioncapture
import numpy as np

from .convert import quat_to_rpy, unwrap_yaw


class ViconThread(threading.Thread):
    def __init__(self, hostname, object_name, cf=None, feed_ekf=True):
        super().__init__(daemon=True)
        self.hostname, self.object_name = hostname, object_name
        self.cf, self.feed_ekf = cf, feed_ekf and cf is not None

        self._running = True
        self._lock = threading.Lock()
        self._pose = None          # (t, pos(3), rpy(3)) with yaw unwrapped
        self._yaw_prev = 0.0
        self.n_frames = 0
        self.n_missing = 0

    def run(self):
        mc = motioncapture.connect("vicon", {"hostname": self.hostname})
        while self._running:
            mc.waitForNextFrame()
            if self.object_name not in mc.rigidBodies:
                self.n_missing += 1
                continue
            ob = mc.rigidBodies[self.object_name]
            pos = np.asarray(ob.position, dtype=float)
            q = ob.rotation                      # x, y, z, w

            if self.feed_ekf:
                try:
                    self.cf.extpos.send_extpose(pos[0], pos[1], pos[2],
                                                q.x, q.y, q.z, q.w)
                except Exception:
                    pass                          # never kill the thread on radio hiccups

            roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
            with self._lock:
                self._yaw_prev = unwrap_yaw(yaw, self._yaw_prev)
                self._pose = (time.time(), pos,
                              np.array([roll, pitch, self._yaw_prev]))
                self.n_frames += 1

    def latest(self):
        """(t, pos, rpy) or None. Non-blocking."""
        with self._lock:
            return self._pose

    def wait_for_pose(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest() is not None:
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self._running = False
