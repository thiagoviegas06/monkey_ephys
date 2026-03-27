import os
import glob
from pathlib import Path
import numpy as np

class SessionDataPhase2:
    """Class to hold session data and paths. Works for Phase 2 and Phase 1."""
    def __init__(self, data_path, is_train=True):
        self.data_path = data_path
        self.is_train = is_train
        
        split_dir = "train" if is_train else "test"
        self.split_path = os.path.join(data_path, split_dir)
        
        # Find all session IDs by looking at sbp files
        sbp_files = glob.glob(os.path.join(self.split_path, "*_sbp.npy"))
        self.sessions = sorted(list(set([Path(f).stem.split('_')[0] for f in sbp_files])))
        print(f"Found {len(self.sessions)} sessions in {self.split_path}.")
        
    def generate_session_obj(self, source_name="phase2"):
        session_objects = []
        for session_id in self.sessions:
            session_objects.append(SessionObjPhase2(self.data_path, session_id, self.is_train, source_name=source_name))
        return session_objects

class SessionObjPhase2:
    def __init__(self, data_path, session_id, is_train=True, source_name="phase2"):
        self.data_path = data_path
        self.session_id = session_id
        self.is_train = is_train
        self.split_dir = "train" if is_train else "test"
        self.source_name = source_name
        
    def get_sbp_path(self):
        return os.path.join(self.data_path, self.split_dir, f"{self.session_id}_sbp.npy")
        
    def get_kin_path(self):
        return os.path.join(self.data_path, self.split_dir, f"{self.session_id}_kinematics.npy")
        
    def load_data(self):
        sbp_path = self.get_sbp_path()
        try:
            sbp = np.load(sbp_path).astype(np.float32)
            kinematics = None
            if self.is_train:
                kin_path = self.get_kin_path()
                if os.path.exists(kin_path):
                    kinematics = np.load(kin_path).astype(np.float32)
            return sbp, kinematics
        except Exception as e:
            print(f"Error loading data for session {self.session_id}: {e}")
            return None, None
