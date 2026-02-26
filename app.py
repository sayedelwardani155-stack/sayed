from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re
from urllib.parse import urlparse, unquote
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

# ==================== SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)


# ==================== INPUT VALIDATION UTILITIES ====================
class InputValidator:
    """Validate and sanitize user inputs to prevent injection attacks"""

    MAX_URL_LENGTH = 2048
    MAX_CONTENT_LENGTH = 10000
    MAX_FEATURES_LENGTH = 100

    # Whitelist of allowed characters in URLs
    ALLOWED_URL_CHARS = re.compile(r'^[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]*$')

    @staticmethod
    def validate_url(url):
        """
        Validate URL format and length
        Returns: (is_valid, sanitized_url)
        """
        if not url or not isinstance(url, str):
            return False, ""

        # Check length
        if len(url) > InputValidator.MAX_URL_LENGTH:
            logger.warning(f"URL exceeds max length: {len(url)} > {InputValidator.MAX_URL_LENGTH}")
            return False, ""

        try:
            # Parse URL to validate structure
            parsed = urlparse(url)

            # Reconstruct URL from parsed components
            sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                sanitized_url += f"?{parsed.query}"
            if parsed.fragment:
                sanitized_url += f"#{parsed.fragment}"

            return True, sanitized_url
        except Exception as e:
            logger.warning(f"Invalid URL format: {str(e)}")
            return False, ""

    @staticmethod
    def validate_content(content):
        """
        Validate content/body data
        Returns: (is_valid, sanitized_content)
        """
        if not content:
            return True, ""

        if not isinstance(content, str):
            return False, ""

        if len(content) > InputValidator.MAX_CONTENT_LENGTH:
            logger.warning(f"Content exceeds max length: {len(content)} > {InputValidator.MAX_CONTENT_LENGTH}")
            return False, ""

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t\r')
        return True, sanitized

    @staticmethod
    def validate_method(method):
        """
        Validate HTTP method
        Returns: (is_valid, method)
        """
        valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}
        if not method or not isinstance(method, str):
            return False, ""

        method_upper = method.upper()
        if method_upper not in valid_methods:
            logger.warning(f"Invalid HTTP method: {method}")
            return False, ""

        return True, method_upper

    @staticmethod
    def validate_features(features):
        """
        Validate ML features list
        Returns: (is_valid, features)
        """
        if features is None:
            return True, None

        if not isinstance(features, list):
            return False, None

        if len(features) > InputValidator.MAX_FEATURES_LENGTH:
            logger.warning(f"Features list exceeds max length")
            return False, None

        # Validate each feature is numeric
        try:
            validated_features = []
            for feature in features:
                if isinstance(feature, (int, float)):
                    # Clamp extreme values
                    clamped = max(-1e6, min(1e6, float(feature)))
                    validated_features.append(clamped)
                else:
                    return False, None
            return True, validated_features
        except Exception as e:
            logger.warning(f"Invalid features format: {str(e)}")
            return False, None


# ==================== ATTACK PATTERNS WITH REGEX ====================
class AttackPatternDetector:
    """Detect attack patterns using compiled regex for performance and security"""

    def __init__(self):
        # Compile regex patterns once for performance
        self.patterns = {
            'SQL_INJECTION': {
                'regex': self._compile_sql_injection_patterns(),
                'description': 'SQL Injection Attack',
                'riskLevel': 'CRITICAL'
            },
            'XSS': {
                'regex': self._compile_xss_patterns(),
                'description': 'Cross-Site Scripting (XSS)',
                'riskLevel': 'CRITICAL'
            },
            'DIRECTORY_TRAVERSAL': {
                'regex': self._compile_directory_traversal_patterns(),
                'description': 'Directory Traversal Attack',
                'riskLevel': 'HIGH'
            },
            'COMMAND_INJECTION': {
                'regex': self._compile_command_injection_patterns(),
                'description': 'Command Injection Attack',
                'riskLevel': 'CRITICAL'
            }
        }

    @staticmethod
    def _compile_sql_injection_patterns():
        """
        Compile SQL injection patterns with word boundaries and case-insensitive matching
        Avoids false positives by using word boundaries
        """
        patterns = [
            r'\bOR\b\s+\d+\s*=\s*\d+',  # OR 1=1
            r'\bDROP\b\s+\b(?:TABLE|DATABASE|SCHEMA)\b',  # DROP TABLE
            r'\bUNION\b\s+\bSELECT\b',  # UNION SELECT
            r'--\s*$',  # SQL comment at end
            r'/\*.*?\*/',  # Multi-line comment
            r'\bEXEC\b\s*\(',  # EXEC(
            r'\bEXECUTE\b\s*\(',  # EXECUTE(
            r"'\s*\bOR\b\s*'",  # ' OR '
            r'\d+\s*\bAND\b\s*\d+\s*=\s*\d+',  # AND condition
            r'\bINSERT\s+INTO\b',  # INSERT INTO
            r'\bDELETE\s+FROM\b',  # DELETE FROM
            r'\bUPDATE\b.*\bSET\b',  # UPDATE ... SET
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        return re.compile(combined, re.IGNORECASE)

    @staticmethod
    def _compile_xss_patterns():
        """
        Compile XSS patterns with proper escaping
        """
        patterns = [
            r'<\s*script[^>]*>',  # <script>
            r'javascript\s*:',  # javascript:
            r'on(?:load|error|click|focus|blur|change|submit|reset)\s*=',  # Event handlers
            r'<\s*iframe[^>]*>',  # <iframe>
            r'<\s*embed[^>]*>',  # <embed>
            r'<\s*object[^>]*>',  # <object>
            r'alert\s*\(',  # alert(
            r'eval\s*\(',  # eval(
            r'<\s*svg[^>]*>',  # <svg>
            r'<\s*img[^>]*>.*?on\w+\s*=',  # <img with event handler>
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        return re.compile(combined, re.IGNORECASE)

    @staticmethod
    def _compile_directory_traversal_patterns():
        """
        Compile directory traversal patterns with URL decoding awareness
        """
        patterns = [
            r'\.{2}[/\\]',  # ../
            r'%2e%2e',  # URL encoded ../
            r'%252e%252e',  # Double encoded
            r'etc/passwd',  # /etc/passwd
            r'windows/system32',  # Windows path
            r'winnt/system32',  # Windows alternate
            r'/root\b',  # /root directory
            r'/admin\b',  # /admin directory
            r'boot\.ini',  # Windows boot file
            r'config\.sys',  # Windows config file
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        return re.compile(combined, re.IGNORECASE)

    @staticmethod
    def _compile_command_injection_patterns():
        """
        Compile command injection patterns
        """
        patterns = [
            r';\s*(?:ls|cat|rm|wget|curl|bash|sh)\b',  # ; command
            r'\|\s*(?:nc|ncat|bash|sh|cat)',  # Pipe to command
            r'`[^`]*`',  # Backtick execution
            r'\$\([^)]*\)',  # $(command)
            r'&&\s*(?:rm|del|wget)',  # && command
            r'\|\|\s*(?:nc|bash)',  # || command
            r'>\s*/dev/',  # Redirect to /dev/
            r'</dev/null',  # Redirect from /dev/null
            r'/bin/(?:bash|sh|zsh)',  # Shell paths
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        return re.compile(combined, re.IGNORECASE)

    def detect(self, url, content, method):
        """
        Detect attack patterns in URL and content
        Returns: (detected_attack_type, confidence_score, matched_pattern)
        """
        # Decode URL-encoded content to catch encoded attacks
        try:
            decoded_url = unquote(url)
            decoded_content = unquote(content) if content else ""
        except Exception as e:
            logger.warning(f"URL decoding error: {str(e)}")
            decoded_url = url
            decoded_content = content or ""

        combined_text = f"{decoded_url} {decoded_content}".lower()

        # Check each pattern type
        for attack_type, pattern_data in self.patterns.items():
            match = pattern_data['regex'].search(combined_text)
            if match:
                confidence = 0.85  # High confidence for regex matches
                logger.warning(f"Attack pattern detected: {attack_type} in {url}")
                return attack_type, confidence, pattern_data

        return None, 0.0, None


# ==================== DATASET LOADER ====================
def load_dataset(csv_path='dataset.csv'):
    """Load the dataset safely"""
    if not os.path.exists(csv_path):
        logger.warning(f"Dataset not found at {csv_path}. Starting with empty model.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded successfully: {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {str(e)}")
        return pd.DataFrame()


# ==================== AI ANOMALY DETECTOR ====================
class AnomalyDetector:
    def __init__(self, dataset_df=None):
        self.total_requests = 0
        self.anomalies_detected = 0
        self.normal_requests = 0
        self.model_accuracy = 0.0
        self.request_history = []
        self.attack_detector = AttackPatternDetector()

        # ML Model components
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.model_trained = False
        self.feature_columns = []

        if dataset_df is not None and not dataset_df.empty:
            self.train_on_dataset(dataset_df)

    def preprocess_data(self, df):
        """Extract and clean numeric features for the ML model"""
        df_clean = df.copy().fillna(0)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        if 'is_anomaly' in numeric_cols:
            numeric_cols.remove('is_anomaly')

        self.feature_columns = numeric_cols
        return df_clean[numeric_cols] if numeric_cols else pd.DataFrame()

    def train_on_dataset(self, df):
        """Train the Isolation Forest model"""
        try:
            X = self.preprocess_data(df)
            if X.empty:
                logger.warning("No valid numeric features found for training.")
                return

            # Scale data
            X_scaled = self.scaler.fit_transform(X)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(X_scaled)
            self.model_trained = True

            # Calculate accuracy
            predictions = self.isolation_forest.predict(X_scaled)
            anomaly_count = sum(1 for p in predictions if p == -1)
            self.model_accuracy = (len(X) - anomaly_count) / len(X) if len(X) > 0 else 0.0

            logger.info(f"✅ ML Model trained! Features: {self.feature_columns}")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.model_trained = False

    def predict(self, method, url, content='', features=None):
        """
        Evaluate a request using ML (if available) or Rules

        Args:
            method: HTTP method (validated)
            url: Request URL (validated and sanitized)
            content: Request body/content (validated and sanitized)
            features: Optional ML features list (validated)
        """
        self.total_requests += 1
        confidence = 0.0
        is_anomaly = False
        attack_type = None
        model_used = 'Pattern-based'
        detected_attack = None

        # 1. Rule-based detection using secure regex patterns
        detected_attack_type, pattern_confidence, attack_data = self.attack_detector.detect(url, content, method)

        if detected_attack_type:
            is_anomaly = True
            confidence = pattern_confidence
            attack_type = detected_attack_type
            detected_attack = attack_data
            model_used = 'Rule-based (Regex)'

        # 2. ML detection if model is trained and patterns didn't trigger
        if not is_anomaly and self.model_trained and features is not None:
            if isinstance(features, list) and len(features) == len(self.feature_columns):
                try:
                    X_scaled = self.scaler.transform([features])
                    prediction = self.isolation_forest.predict(X_scaled)[0]
                    is_anomaly = (prediction == -1)

                    if is_anomaly:
                        raw_score = self.isolation_forest.score_samples(X_scaled)[0]
                        confidence = min(abs(raw_score), 0.99)
                        model_used = 'ML-based (Isolation Forest)'
                except Exception as e:
                    logger.warning(f"ML prediction error: {e}")

        # Update metrics
        if is_anomaly:
            self.anomalies_detected += 1
        else:
            self.normal_requests += 1

        # Format output
        result = {
            'prediction_code': 1 if is_anomaly else 0,
            'confidence': round(confidence, 3),
            'attack_type': attack_type if is_anomaly else 'None',
            'url': url,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_used
        }

        # Save to history (keep last 1000)
        self.request_history.append(result)
        if len(self.request_history) > 1000:
            self.request_history.pop(0)

        status_text = '🔴 ANOMALY' if is_anomaly else '🟢 NORMAL'
        logger.info(f"{method} {url} -> {status_text} ({model_used}, Conf: {confidence:.2f})")

        return result


# ==================== INITIALIZATION ====================
detector = AnomalyDetector(load_dataset('dataset.csv'))


# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/detect', methods=['POST'])
def detect_anomaly():
    """
    Detect anomalies in HTTP requests

    Expects JSON:
    {
        "Method": "GET|POST|PUT|DELETE|PATCH",
        "URL": "/path/to/resource?query=value",
        "content": "request body (optional)",
        "features": [1.0, 2.5, 3.2, ...] (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logger.warning("Invalid or missing JSON payload")
            return jsonify({'error': 'Invalid JSON payload'}), 400

        # Extract and validate inputs
        raw_method = data.get('Method', 'GET')
        raw_url = data.get('URL', '/')
        raw_content = data.get('content', '')
        raw_features = data.get('features', None)

        # Validate all inputs
        is_valid_method, method = InputValidator.validate_method(raw_method)
        if not is_valid_method:
            logger.warning(f"Invalid method: {raw_method}")
            return jsonify({'error': 'Invalid HTTP method'}), 400

        is_valid_url, url = InputValidator.validate_url(raw_url)
        if not is_valid_url:
            logger.warning(f"Invalid URL: {raw_url}")
            return jsonify({'error': 'Invalid URL format'}), 400

        is_valid_content, content = InputValidator.validate_content(raw_content)
        if not is_valid_content:
            logger.warning(f"Invalid content")
            return jsonify({'error': 'Invalid content format'}), 400

        is_valid_features, features = InputValidator.validate_features(raw_features)
        if not is_valid_features:
            logger.warning(f"Invalid features")
            return jsonify({'error': 'Invalid features format'}), 400

        # Perform detection with validated inputs
        result = detector.predict(method, url, content, features)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"/detect Error: {str(e)}")
        return jsonify({'error': 'Processing error', 'details': 'An unexpected error occurred'}), 500


@app.route('/api/reload-dataset', methods=['POST'])
def reload_dataset():
    try:
        data = request.get_json() or {}
        csv_path = data.get('csv_path', 'dataset.csv')

        # Validate path to prevent directory traversal
        if '..' in csv_path or csv_path.startswith('/'):
            logger.warning(f"Suspicious csv_path: {csv_path}")
            return jsonify({'error': 'Invalid file path'}), 400

        df = load_dataset(csv_path)
        if df.empty:
            return jsonify({'error': 'Dataset is empty or invalid'}), 400

        detector.train_on_dataset(df)

        return jsonify({
            'status': 'success',
            'records_loaded': len(df),
            'model_trained': detector.model_trained,
            'feature_columns': detector.feature_columns
        }), 200

    except Exception as e:
        logger.error(f"Dataset reload error: {str(e)}")
        return jsonify({'error': 'Failed to reload dataset'}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    rate = (detector.anomalies_detected / detector.total_requests * 100) if detector.total_requests > 0 else 0
    return jsonify({
        'total_requests': detector.total_requests,
        'anomalies_detected': detector.anomalies_detected,
        'normal_requests': detector.normal_requests,
        'detection_rate': round(rate, 2),
        'model_trained': detector.model_trained
    }), 200


@app.route('/api/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 50, type=int)
    # Validate limit to prevent abuse
    limit = min(limit, 500)  # Max 500 entries
    limit = max(limit, 1)  # Min 1 entry
    return jsonify({'history': detector.request_history[-limit:]}), 200


@app.route('/api/patterns', methods=['GET'])
def get_patterns():
    """Get attack pattern statistics from history"""
    stats_dict = {
        'SQL_INJECTION': {'description': 'SQL Injection Attack', 'riskLevel': 'CRITICAL', 'count': 0},
        'XSS': {'description': 'Cross-Site Scripting (XSS)', 'riskLevel': 'CRITICAL', 'count': 0},
        'DIRECTORY_TRAVERSAL': {'description': 'Directory Traversal Attack', 'riskLevel': 'HIGH', 'count': 0},
        'COMMAND_INJECTION': {'description': 'Command Injection Attack', 'riskLevel': 'CRITICAL', 'count': 0}
    }

    for req in detector.request_history:
        atk = req.get('attack_type')
        if atk and atk in stats_dict:
            stats_dict[atk]['count'] += 1

    return jsonify(stats_dict), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== RUN ====================
if __name__ == '__main__':
    print("🚀 AI Anomaly Detection Backend Started (Security Hardened)")
    print(f"📊 ML Model Trained: {detector.model_trained}")
    print(f"🔒 Input validation enabled")
    print(f"🔐 Regex-based pattern detection enabled")
    app.run(debug=False, host='127.0.0.1', port=5000)