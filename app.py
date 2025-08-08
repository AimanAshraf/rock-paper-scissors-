from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import json
from threading import Lock
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/api/health")
def health():
    return {"ok": True}


# Global game state
game_state = {
    'player_score': 0,
    'computer_score': 0,
    'current_round': 1,
    'game_status': 'waiting',  # waiting, countdown, playing, result, game_over, paused
    'countdown': 3,
    'player_move': None,
    'computer_move': None,
    'last_winner': None,
    'hand_detected': False,
    'is_game_active': False,
    'paused_state': None
}

game_lock = Lock()

# MediaPipe setup
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
except Exception as e:
    print(f"MediaPipe initialization error: {e}")
    # Fallback initialization
    hands = None

class GestureRecognizer:
    def __init__(self):
        pass
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture from hand landmarks"""
        if not landmarks or len(landmarks) != 21:
            return 'unknown'
        
        # Get finger states (extended or not)
        finger_states = self._get_finger_states(landmarks)
        
        # Rock: All fingers closed except thumb
        if self._is_rock(finger_states):
            return 'rock'
        
        # Paper: All fingers extended
        if self._is_paper(finger_states):
            return 'paper'
        
        # Scissors: Index and middle fingers extended, others closed
        if self._is_scissors(finger_states):
            return 'scissors'
        
        return 'unknown'
    
    def _get_finger_states(self, landmarks):
        """Determine which fingers are extended"""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # Previous joints
        
        finger_states = []
        
        # Thumb (compare x coordinates)
        thumb_tip = landmarks[finger_tips[0]]
        thumb_pip = landmarks[finger_pips[0]]
        finger_states.append(abs(thumb_tip.x - thumb_pip.x) > 0.04)
        
        # Other fingers (compare y coordinates)
        for i in range(1, 5):
            tip = landmarks[finger_tips[i]]
            pip = landmarks[finger_pips[i]]
            finger_states.append(tip.y < pip.y)  # Extended if tip is above pip
        
        return finger_states
    
    def _is_rock(self, finger_states):
        """Check if gesture is rock (all fingers closed except possibly thumb)"""
        return not any(finger_states[1:])  # Index, middle, ring, pinky all closed
    
    def _is_paper(self, finger_states):
        """Check if gesture is paper (all fingers extended)"""
        return all(finger_states)
    
    def _is_scissors(self, finger_states):
        """Check if gesture is scissors (index and middle extended, others closed)"""
        return finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]
    
    def determine_winner(self, player_move, computer_move):
        """Determine winner based on moves"""
        if player_move == computer_move:
            return 'tie'
        
        winning_combinations = {
            ('rock', 'scissors'): 'player',
            ('paper', 'rock'): 'player',
            ('scissors', 'paper'): 'player'
        }
        
        return winning_combinations.get((player_move, computer_move), 'computer')
    
    def generate_computer_move(self):
        """Generate random computer move"""
        return random.choice(['rock', 'paper', 'scissors'])

gesture_recognizer = GestureRecognizer()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/game_state')
def get_game_state():
    with game_lock:
        return jsonify(game_state)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game_state
    with game_lock:
        game_state.update({
            'player_score': 0,
            'computer_score': 0,
            'current_round': 1,
            'game_status': 'waiting',
            'countdown': 3,
            'player_move': None,
            'computer_move': None,
            'last_winner': None,
            'hand_detected': False,
            'is_game_active': False,
            'paused_state': None
        })
    return jsonify({'status': 'success'})

@app.route('/api/pause_game', methods=['POST'])
def pause_game():
    global game_state
    with game_lock:
        if game_state['game_status'] not in ['waiting', 'game_over']:
            game_state['paused_state'] = {
                'game_status': game_state['game_status'],
                'countdown': game_state['countdown'],
                'player_move': game_state['player_move'],
                'computer_move': game_state['computer_move'],
                'hand_detected': game_state['hand_detected'],
                'is_game_active': game_state['is_game_active']
            }
            game_state['game_status'] = 'paused'
            game_state['is_game_active'] = False
    return jsonify({'status': 'success'})

@app.route('/api/resume_game', methods=['POST'])
def resume_game():
    global game_state
    with game_lock:
        if game_state['paused_state']:
            paused = game_state['paused_state']
            game_state.update({
                'game_status': paused['game_status'],
                'countdown': paused['countdown'],
                'player_move': paused['player_move'],
                'computer_move': paused['computer_move'],
                'hand_detected': paused['hand_detected'],
                'is_game_active': paused['is_game_active']
            })
            game_state['paused_state'] = None
    return jsonify({'status': 'success'})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global game_state
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        detected_gesture = 'unknown'
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            detected_gesture = gesture_recognizer.recognize_gesture(hand_landmarks.landmark)
        
        with game_lock:
            game_state['hand_detected'] = hand_detected
            
            # Update player move if in active game phases
            if detected_gesture != 'unknown' and game_state['game_status'] in ['waiting', 'countdown', 'playing']:
                game_state['player_move'] = detected_gesture
            
            # Auto-start countdown when hand detected
            if hand_detected and game_state['game_status'] == 'waiting':
                game_state['game_status'] = 'countdown'
                game_state['countdown'] = 3
                game_state['is_game_active'] = True
        
        return jsonify({
            'hand_detected': hand_detected,
            'gesture': detected_gesture,
            'confidence': 90 if detected_gesture != 'unknown' else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/countdown_tick', methods=['POST'])
def countdown_tick():
    global game_state
    
    with game_lock:
        if game_state['game_status'] == 'countdown' and game_state['countdown'] > 0:
            game_state['countdown'] -= 1
            
            if game_state['countdown'] == 0:
                game_state['game_status'] = 'playing'
                
                # After 2 seconds of playing, determine result
                def determine_result():
                    time.sleep(2)
                    with game_lock:
                        if game_state['game_status'] == 'playing':
                            computer_move = gesture_recognizer.generate_computer_move()
                            player_move = game_state['player_move'] or 'rock'
                            
                            game_state['computer_move'] = computer_move
                            game_state['player_move'] = player_move
                            
                            winner = gesture_recognizer.determine_winner(player_move, computer_move)
                            game_state['last_winner'] = winner
                            
                            # Update scores
                            if winner == 'player':
                                game_state['player_score'] += 1
                            elif winner == 'computer':
                                game_state['computer_score'] += 1
                            
                            # Check for game over
                            if game_state['player_score'] >= 5 or game_state['computer_score'] >= 5:
                                game_state['game_status'] = 'game_over'
                            else:
                                game_state['game_status'] = 'result'
                                game_state['current_round'] += 1
                                
                                # Auto-reset after 3 seconds
                                def reset_round():
                                    time.sleep(3)
                                    with game_lock:
                                        if game_state['game_status'] == 'result':
                                            game_state.update({
                                                'game_status': 'waiting',
                                                'countdown': 3,
                                                'player_move': None,
                                                'computer_move': None,
                                                'hand_detected': False,
                                                'is_game_active': False
                                            })
                                
                                import threading
                                threading.Thread(target=reset_round, daemon=True).start()
                
                import threading
                threading.Thread(target=determine_result, daemon=True).start()
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)