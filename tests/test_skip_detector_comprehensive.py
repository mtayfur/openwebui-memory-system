"""
Comprehensive SkipDetector Testing Suite
Tests all three layers: Size Validation, Fast-Path Detection, and Semantic Classification
"""

import sys
import logging
from typing import Dict, List, Tuple, Optional, Sequence
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '/Users/tayfur/Workspace/code/personal/openwebui-memory-system')

from memory_system import SkipDetector, Constants

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Also use print for critical output
def log_and_print(message):
    """Helper to both log and print."""
    logger.info(message)
    print(message, flush=True)


class TestDataset:
    """Comprehensive test dataset covering all skip categories and conversational cases."""
    
    # Layer 1: Size Validation Tests
    SIZE_TESTS = [
        # Too short
        ("Hi", "SKIP_SIZE", "too_short"),
        ("OK", "SKIP_SIZE", "too_short"),
        ("Yes", "SKIP_SIZE", "too_short"),
        ("", "SKIP_SIZE", "empty"),
        ("   ", "SKIP_SIZE", "whitespace_only"),
        
        # Too long (>2500 chars)
        ("This is a very long message. " * 100, "SKIP_SIZE", "too_long"),
        ("Lorem ipsum dolor sit amet. " * 150, "SKIP_SIZE", "too_long"),
    ]
    
    # Layer 2: Fast-Path Detection Tests
    FAST_PATH_TESTS = [
        # Pattern 1: Multiple URLs (various formats)
        (
            "Check these links: https://example1.com https://example2.com https://example3.com https://example4.com https://example5.com https://example6.com",
            "SKIP_TECHNICAL",
            "multiple_urls_basic"
        ),
        (
            "Resources: http://api.github.com http://stackoverflow.com https://developer.mozilla.org https://docs.python.org http://npmjs.com https://golang.org",
            "SKIP_TECHNICAL",
            "multiple_urls_mixed"
        ),
        (
            "Documentation links:\nhttps://reactjs.org/docs\nhttps://vuejs.org/guide\nhttps://angular.io/docs\nhttps://svelte.dev/tutorial\nhttp://expressjs.com/en/guide\nhttps://fastapi.tiangolo.com",
            "SKIP_TECHNICAL",
            "multiple_urls_multiline"
        ),
        
        # Pattern 2: Long alphanumeric tokens (hashes, base64, keys)
        (
            "Here is the token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",
            "SKIP_TECHNICAL",
            "long_token_jwt"
        ),
        (
            "API key: sk_live_51HzR2pKZvKYlo2C9x7yT3kP8qN4mL6jW5vU2hF1gD8sA7cB6eX9wQ0rT5yU4iO3pL2kM8nJ7hG6fE5dC4bA3zS2xW1vU0tR9qP8oN7mL6kJ5iH4gF3eD2cB1aZ0yX9wV8uT7sR6qP5oN4mL3kJ2iH1gF0eD9cC8bA7zZ6yY5xW4vU3tS2rQ1pP0oN9mM8lL7kK6jJ5iI4hH3gG2fF1eE0dD9cC8bB7aA6zZ5yY4xX3wW2vV1uU0tT9sS8rR7qQ6pP5oO4nN3mM2lL1kK0jJ9iI8hH7gG6fF5eE4dD3cC2bB1aA0",
            "SKIP_TECHNICAL",
            "long_token_api_key"
        ),
        (
            "Commit hash: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2",
            "SKIP_TECHNICAL",
            "long_token_hash"
        ),
        (
            "Session: dGhpcyBpcyBhIHZlcnkgbG9uZyBiYXNlNjQgZW5jb2RlZCBzdHJpbmcgdGhhdCBzaG91bGQgYmUgZGV0ZWN0ZWQgYXMgdGVjaG5pY2FsIGNvbnRlbnQ",
            "SKIP_TECHNICAL",
            "long_token_base64"
        ),
        
        # Pattern 3: Markdown/text separators (various patterns)
        (
            "Section 1\n---\n---\n---\nContent here\n===\n===\n===\nMore content",
            "SKIP_TECHNICAL",
            "markdown_separators_mixed"
        ),
        (
            "Chapter One\n___________\nContent\n___________\nChapter Two\n___________",
            "SKIP_TECHNICAL",
            "markdown_separators_underscores"
        ),
        (
            "Header\n***************\nBody text\n***************\nFooter",
            "SKIP_TECHNICAL",
            "markdown_separators_asterisks"
        ),
        
        # Pattern 4: Command-line patterns (diverse commands)
        (
            "$ npm install express\n$ git clone repo\n$ docker build -t image\n$ pip install requirements",
            "SKIP_TECHNICAL",
            "command_lines_basic"
        ),
        (
            "$ sudo apt-get update\n$ sudo apt-get install nginx\n$ systemctl start nginx\n$ curl localhost:80",
            "SKIP_TECHNICAL",
            "command_lines_system"
        ),
        (
            "$ kubectl get pods\n$ kubectl apply -f deployment.yaml\n$ kubectl logs pod-name\n$ kubectl describe service",
            "SKIP_TECHNICAL",
            "command_lines_kubernetes"
        ),
        (
            "# install dependencies\n# build the project\n# run tests\n# deploy to production",
            "SKIP_TECHNICAL",
            "command_lines_comments"
        ),
        (
            "$ cargo build --release\n$ cargo test\n$ cargo run --bin myapp\n$ cargo clippy",
            "SKIP_TECHNICAL",
            "command_lines_rust"
        ),
        
        # Pattern 5: High path/URL density (various operating systems)
        (
            "Path: /usr/local/bin/python3 /home/user/projects/app/src/main.py /var/log/system.log",
            "SKIP_TECHNICAL",
            "path_density_unix"
        ),
        (
            "Files: C:\\Program Files\\App\\bin\\app.exe C:\\Users\\John\\Documents\\file.txt C:\\Windows\\System32\\config.ini",
            "SKIP_TECHNICAL",
            "path_density_windows"
        ),
        (
            "Check /etc/nginx/nginx.conf and /var/www/html/index.html then /usr/share/doc/readme.txt",
            "SKIP_TECHNICAL",
            "path_density_config"
        ),
        (
            "Import paths: ./src/components/Button.tsx ../utils/helpers.js ../../config/database.json ./styles/main.css",
            "SKIP_TECHNICAL",
            "path_density_relative"
        ),
        
        # Pattern 6: Markup character density (various formats)
        (
            '{"key": "value", "nested": {"array": [1, 2, 3]}, "more": {"data": true}}',
            "SKIP_TECHNICAL",
            "markup_density_json"
        ),
        (
            '<root><item id="1"><name>Test</name><value>100</value></item><item id="2"><name>Data</name></item></root>',
            "SKIP_TECHNICAL",
            "markup_density_xml"
        ),
        (
            "[{id: 1, name: 'Alice'}, {id: 2, name: 'Bob'}, {id: 3, name: 'Charlie'}]",
            "SKIP_TECHNICAL",
            "markup_density_array"
        ),
        (
            "config: {server: {host: '0.0.0.0', port: 8080}, database: {name: 'mydb', user: 'admin'}}",
            "SKIP_TECHNICAL",
            "markup_density_config"
        ),
        
        # Pattern 7: Highly structured multi-line content
        (
            "Line 1\n  Indented\n    More indent\nLine 2\n  Nested: value\n    Another: data\nLine 3\n  Key: pair\nLine 4\n  More: structure\nLine 5\n  Even: more\nLine 6\n  Continuing: on\nLine 7\n  Still: going\nLine 8\n  Data: here\nLine 9\n  Final: item",
            "SKIP_TECHNICAL",
            "structured_multiline_nested"
        ),
        (
            "Root:\n  Child1:\n    Property: value\n    Items:\n      - item1\n      - item2\n  Child2:\n    Data: content\n    More:\n      - entry1\n      - entry2\n  Child3:\n    Final: section",
            "SKIP_TECHNICAL",
            "structured_multiline_yaml_like"
        ),
        (
            "Section A\n  Subsection 1\n    Detail: info\n  Subsection 2\n    Detail: data\nSection B\n  Subsection 3\n    Detail: content\n  Subsection 4\n    Detail: value\nSection C\n  Subsection 5\n    Detail: text",
            "SKIP_TECHNICAL",
            "structured_multiline_hierarchical"
        ),
        
        # Pattern 8: Code-like indentation (multiple languages)
        (
            "def calculate():\n    result = 0\n    for i in range(10):\n        result += i\n    return result",
            "SKIP_TECHNICAL",
            "code_indentation_python"
        ),
        (
            "function process() {\n    let count = 0;\n    for (let i = 0; i < 10; i++) {\n        count += i;\n    }\n    return count;\n}",
            "SKIP_TECHNICAL",
            "code_indentation_javascript"
        ),
        (
            "class Calculator {\n    public int add(int a, int b) {\n        int result = a + b;\n        return result;\n    }\n}",
            "SKIP_TECHNICAL",
            "code_indentation_java"
        ),
        (
            "impl Calculator {\n    fn multiply(&self, a: i32, b: i32) -> i32 {\n        let result = a * b;\n        return result;\n    }\n}",
            "SKIP_TECHNICAL",
            "code_indentation_rust"
        ),
        
        # Pattern 9: High special character ratio (various types)
        (
            "!@#$%^&*()_+-=[]{}|;:',.<>?/~`!@#$%^&*()_+-=[]{}|;:',.<>?/~`!@#$%^&*()",
            "SKIP_TECHNICAL",
            "special_chars_symbols"
        ),
        (
            "======================================>>>>>>>>>><<<<<<<<<<<||||||||||||",
            "SKIP_TECHNICAL",
            "special_chars_ascii_art"
        ),
        (
            "***###***###***###***###***###***###***###***###***###***###***###***",
            "SKIP_TECHNICAL",
            "special_chars_pattern"
        ),
        
        # Pattern 10: Mixed technical patterns
        (
            "Config file at /etc/app/config.json contains: {\"api\": \"https://api.example.com\", \"timeout\": 5000}",
            "SKIP_TECHNICAL",
            "mixed_path_json"
        ),
        (
            "Run: $ curl https://api.github.com/users/octocat | jq '.login'",
            "SKIP_TECHNICAL",
            "mixed_command_url"
        ),
        (
            "Error in /var/log/app.log: Exception at line 42\n---\nStack trace: func1() -> func2() -> func3()",
            "SKIP_TECHNICAL",
            "mixed_path_error"
        ),
        
        # LONG MULTI-SENTENCE FAST-PATH TESTS
        (
            "I found several useful resources for this project. Check out https://docs.python.org/3/tutorial/ for Python basics, then review https://flask.palletsprojects.com/en/2.3.x/ for web framework documentation. You should also look at https://www.sqlalchemy.org/ for database ORM, https://pytest.org/ for testing, and https://black.readthedocs.io/ for code formatting. Finally, don't forget https://mypy.readthedocs.io/ for type checking and https://pre-commit.com/ for git hooks.",
            "SKIP_TECHNICAL",
            "long_multiple_urls_paragraph"
        ),
        (
            "To set up the development environment, first run $ npm install to install dependencies. Then execute $ npm run build to compile the TypeScript code. After that, start the development server with $ npm run dev and open another terminal to run $ npm test for the test suite. Finally, you can deploy using $ npm run deploy --production and check the status with $ npm run status.",
            "SKIP_TECHNICAL",
            "long_command_sequence"
        ),
        (
            "The application stores configuration in /etc/myapp/config.yaml and logs to /var/log/myapp/application.log. User data is saved in /home/users/data/profiles.json while cache files go to /tmp/myapp/cache/. The main executable is located at /usr/local/bin/myapp and library files are in /usr/lib/myapp/modules/. Documentation can be found at /usr/share/doc/myapp/README.md and examples are in /opt/myapp/examples/.",
            "SKIP_TECHNICAL",
            "long_path_listing"
        ),
        (
            "Here's the complete API response from the server. It includes nested objects and arrays with multiple levels of data. The JSON structure contains user information, authentication tokens, and metadata. Here it is: {\"status\": \"success\", \"data\": {\"user\": {\"id\": 12345, \"name\": \"John Doe\", \"email\": \"john@example.com\", \"roles\": [\"admin\", \"developer\"]}, \"session\": {\"token\": \"abc123xyz789\", \"expires\": \"2025-12-31\", \"permissions\": [\"read\", \"write\", \"delete\"]}, \"metadata\": {\"timestamp\": \"2025-10-05T14:30:00Z\", \"version\": \"1.2.3\"}}}",
            "SKIP_TECHNICAL",
            "long_json_response"
        ),
        (
            "The deployment process involves several steps with multiple commands and configuration files. First, you need to build the Docker image using the Dockerfile located at ./docker/Dockerfile. Then push it to the registry at https://registry.example.com/myapp:latest. After that, update the Kubernetes deployment manifest at /deployments/k8s/deployment.yaml and apply it with kubectl. The application configuration is stored in /config/production.json and environment variables are defined in .env.production. Logs can be monitored at https://logs.example.com/dashboard and metrics at https://metrics.example.com/grafana.",
            "SKIP_TECHNICAL",
            "long_deployment_mixed"
        ),
        (
            "def process_user_data(user_id, data):\n    \"\"\"Process user data with validation and transformation.\"\"\"\n    try:\n        validated_data = validate_schema(data)\n        transformed = transform_fields(validated_data)\n        result = database.save_user(user_id, transformed)\n        logger.info(f'Processed user {user_id} successfully')\n        return {'status': 'success', 'user_id': user_id, 'data': result}\n    except ValidationError as e:\n        logger.error(f'Validation failed: {e}')\n        raise\n    except DatabaseError as e:\n        logger.critical(f'Database error: {e}')\n        rollback_transaction()\n        raise\n    finally:\n        cleanup_resources()",
            "SKIP_TECHNICAL",
            "long_code_block"
        ),
        
        # REALISTIC USER INTERACTIONS - FAST PATH (Technical Content)
        
        # Sharing code snippets from their project
        (
            """Can you help me understand why this isn't working?
            
function fetchUserData(userId) {
  fetch(`/api/users/${userId}`)
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

I'm getting a CORS error in the console.""",
            "SKIP_TECHNICAL",
            "realistic_code_debug_request"
        ),
        
        # Pasting error logs
        (
            """I'm getting this error and I don't know what it means:

Traceback (most recent call last):
  File "main.py", line 23, in <module>
    result = process_data(input_file)
  File "main.py", line 15, in process_data
    df = pd.read_csv(filename)
  File "/usr/local/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 912, in read_csv
    return _read(filepath_or_buffer, kwds)
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'

What should I do?""",
            "SKIP_TECHNICAL",
            "realistic_error_log_help"
        ),
        
        # Sharing config file
        (
            """Here's my package.json but npm install is failing:

{
  "name": "my-app",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.4.0",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "webpack": "^5.88.0",
    "babel-loader": "^9.1.2",
    "@babel/core": "^7.22.5"
  },
  "scripts": {
    "start": "webpack serve --mode development",
    "build": "webpack --mode production"
  }
}

What's wrong?""",
            "SKIP_TECHNICAL",
            "realistic_config_file_debug"
        ),
        
        # Command line help
        (
            """I'm trying to deploy but getting permission errors:

$ docker push myregistry.azurecr.io/myapp:latest
denied: authentication required
$ kubectl apply -f deployment.yaml
Error from server (Forbidden): error when creating "deployment.yaml": deployments.apps is forbidden
$ az login
You have logged in. Now let us find all the subscriptions to which you have access.

Still not working. Help?""",
            "SKIP_TECHNICAL",
            "realistic_command_line_issues"
        ),
        
        # API response debugging
        (
            """The API is returning this weird response:

{
  "status": 200,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {"field": "email", "issue": "required"},
      {"field": "password", "issue": "min_length"}
    ]
  }
}

But I'm definitely sending email and password in the request body. What's going on?""",
            "SKIP_TECHNICAL",
            "realistic_api_response_debug"
        ),
        
        # Sharing multiple URLs for research
        (
            """I've been researching React performance optimization and found these helpful:

https://react.dev/learn/render-and-commit
https://react.dev/reference/react/memo
https://legacy.reactjs.org/docs/optimizing-performance.html
https://kentcdodds.com/blog/usememo-and-usecallback
https://web.dev/react/
https://www.patterns.dev/posts/react-performance/

Which one should I follow for my use case?""",
            "SKIP_TECHNICAL",
            "realistic_research_urls"
        ),
        
        # Git command history
        (
            """I messed up my git repo. Here's what I did:

$ git add .
$ git commit -m "WIP"
$ git push origin main
$ git pull origin main
Auto-merging src/App.js
CONFLICT (content): Merge conflict in src/App.js
$ git status
On branch main
You have unmerged paths.

Now I don't know how to fix it. Can you help?""",
            "SKIP_TECHNICAL",
            "realistic_git_mess_help"
        ),
        
        # Database query not working
        (
            """This SQL query is returning 0 rows but I know there's data:

SELECT u.name, COUNT(o.id) as order_count
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01'
GROUP BY u.name
HAVING COUNT(o.id) > 10;

When I run SELECT COUNT(*) FROM orders WHERE created_at > '2024-01-01' I get 523 rows.

What am I missing?""",
            "SKIP_TECHNICAL",
            "realistic_sql_query_zero_results"
        ),
        
        # Dockerfile build failing
        (
            """My Docker build keeps failing at this step:

FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]

Error message:
Step 4/8 : RUN npm install
npm ERR! code ENOENT
npm ERR! syscall open
npm ERR! path /app/package.json

Why can't it find package.json when I just copied it?""",
            "SKIP_TECHNICAL",
            "realistic_docker_build_fail"
        ),
        
        # Environment variable issues
        (
            """My app works locally but not in production. Here's my .env file:

DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
REDIS_URL=redis://localhost:6379
API_KEY=sk_test_123456789
JWT_SECRET=my_secret_key_here
NODE_ENV=development
PORT=3000

In production I set these in Heroku config vars but getting "Cannot connect to database". What's different?""",
            "SKIP_TECHNICAL",
            "realistic_env_vars_problem"
        ),
        
        # Regex pattern help
        (
            r"""I need a regex to validate email addresses. I tried this:

/^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$/

But it's not matching emails like john.doe@company.co.uk or user+tag@example.com

What's the correct pattern?""",
            "SKIP_TECHNICAL",
            "realistic_regex_help"
        ),
        
        # Build tool configuration
        (
            r"""My webpack build is failing with this webpack.config.js:

module.exports = {
  entry: './src/index.js',
  output: {
    filename: 'bundle.js',
    path: __dirname + '/dist'
  },
  module: {
    rules: [
      {
        test: /\.jsx?$/,
        exclude: /node_modules/,
        use: 'babel-loader'
      }
    ]
  }
};

Error: Module not found: Error: Can't resolve './App' in '/Users/me/project/src'

The App.js file exists. Why can't webpack find it?""",
            "SKIP_TECHNICAL",
            "realistic_webpack_config"
        ),
        
        # Test debugging
        (
            """My Jest tests are all failing with this error:

FAIL  src/components/Button.test.js
  ● Test suite failed to run
  
    Cannot find module '@testing-library/react' from 'src/components/Button.test.js'
    
    > 1 | import { render, screen } from '@testing-library/react';
        | ^
      2 | import Button from './Button';

I ran npm install @testing-library/react and it's in package.json. What's wrong?""",
            "SKIP_TECHNICAL",
            "realistic_test_failure"
        ),
        
        # Performance issue with logs
        (
            """My Node.js API is really slow. Here's what the logs show:

2024-10-05 14:23:45 INFO  Server started on port 3000
2024-10-05 14:24:12 DEBUG GET /api/users - Query took 3245ms
2024-10-05 14:24:15 DEBUG GET /api/posts - Query took 5678ms
2024-10-05 14:24:18 WARN  Connection pool exhausted, waiting for available connection
2024-10-05 14:24:23 ERROR Database query timeout after 5000ms

SELECT users.*, COUNT(posts.id) as post_count
FROM users
LEFT JOIN posts ON users.id = posts.user_id
GROUP BY users.id;

Should I add indexes? Increase the connection pool size?""",
            "SKIP_TECHNICAL",
            "realistic_performance_logs"
        ),
        
        # Package dependency conflict
        (
            """npm install is giving me dependency conflicts:

npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! 
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@18.2.0
npm ERR! node_modules/react
npm ERR!   react@"^18.2.0" from the root project
npm ERR! 
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" from react-router-dom@5.3.0
npm ERR! node_modules/react-router-dom
npm ERR!   react-router-dom@"^5.3.0" from the root project

How do I fix this? Do I need to downgrade React or upgrade react-router-dom?""",
            "SKIP_TECHNICAL",
            "realistic_dependency_conflict"
        ),
        
        # CSS layout not working
        (
            """This flexbox layout isn't centering properly:

.container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.box {
  width: 300px;
  height: 200px;
  background: blue;
}

The box is centered horizontally but not vertically. The container div is definitely full height in the inspector. What am I missing?""",
            "SKIP_TECHNICAL",
            "realistic_css_layout"
        ),
        
        # Authentication flow debugging
        (
            """My JWT authentication isn't working. Here's my middleware:

const jwt = require('jsonwebtoken');

function authenticateToken(req, res, next) {
  const token = req.headers['authorization'];
  
  if (!token) return res.status(401).send('Access denied');
  
  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).send('Invalid token');
    req.user = user;
    next();
  });
}

Postman request headers:
Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiZW1haWwiOiJ1c2VyQGV4YW1wbGUuY29tIn0.abc123

Getting "Invalid token" error. The token was just generated. What's wrong?""",
            "SKIP_TECHNICAL",
            "realistic_auth_debugging"
        ),
        
        # Git merge conflict
        (
            """I have a git merge conflict and I'm not sure how to resolve it:

<<<<<<< HEAD
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
}
=======
function calculateTotal(items) {
  let total = 0;
  for (let item of items) {
    total += item.price * item.quantity * (1 - item.discount);
  }
  return total;
}
>>>>>>> feature/add-discount-support

Both versions have been tested and work. Which one should I keep? Or should I combine them somehow?""",
            "SKIP_TECHNICAL",
            "realistic_merge_conflict"
        ),
        
        # MORE REALISTIC FAST-PATH SCENARIOS (25 new tests for diversity)
        
        # Mobile app crash logs
        (
            """My iOS app keeps crashing with this error:

*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason: '-[__NSArrayM insertObject:atIndex:]: object cannot be nil'
*** First throw call stack:
(0x1847d9e38 0x183f170f4 0x1846d8584 0x100f2c3a8 0x100f2c1d4 0x186e3a2d8)
libc++abi.dylib: terminating with uncaught exception of type NSException

Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
0   libsystem_kernel.dylib          0x00000001848a0d40 __pthread_kill + 8
1   libsystem_pthread.dylib         0x00000001848d8c20 pthread_kill + 272
2   libsystem_c.dylib               0x000000018481ea90 abort + 140

It happens when users try to add items to their shopping cart. Help?""",
            "SKIP_TECHNICAL",
            "realistic_mobile_crash_log"
        ),
        
        # YAML configuration parsing
        (
            """My Kubernetes deployment won't apply. Here's the YAML:

apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  database.yaml: |
    host: postgres.production.svc.cluster.local
    port: 5432
    database: myapp
    pool:
      min: 2
      max: 10
  redis.yaml: |
    host: redis.production.svc.cluster.local
    port: 6379
    ttl: 3600

Error: error parsing data.database.yaml: yaml: line 4: mapping values are not allowed in this context

What's wrong with my indentation?""",
            "SKIP_TECHNICAL",
            "realistic_yaml_parsing"
        ),
        
        # Browser DevTools console
        (
            """Getting these errors in Chrome DevTools:

[Violation] Added non-passive event listener to a scroll-blocking 'touchstart' event.
Uncaught TypeError: Cannot read properties of undefined (reading 'map')
    at HomePage.jsx:45
    at Array.map (<anonymous>)
    at HomePage (HomePage.jsx:45)
Failed to load resource: the server responded with a status of 404 (Not Found) - styles.css:1
CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.

Which one should I fix first?""",
            "SKIP_TECHNICAL",
            "realistic_browser_console_errors"
        ),
        
        # Linux server logs
        (
            """Server is running slow. Here are the last syslog entries:

Oct  5 14:23:45 web-01 kernel: [12345.678901] Out of memory: Kill process 9876 (node) score 856 or sacrifice child
Oct  5 14:23:46 web-01 systemd[1]: myapp.service: Main process exited, code=killed, status=9/KILL
Oct  5 14:23:47 web-01 systemd[1]: myapp.service: Failed with result 'signal'.
Oct  5 14:23:48 web-01 systemd[1]: myapp.service: Service hold-off time over, scheduling restart.

Memory usage was at 98%. Is this a memory leak?""",
            "SKIP_TECHNICAL",
            "realistic_linux_syslog"
        ),
        
        # GraphQL schema definition
        (
            """Need feedback on my GraphQL schema:

type User {
  id: ID!
  email: String!
  username: String!
  profile: Profile
  posts(first: Int, after: String): PostConnection!
  followers(first: Int): [User!]!
  following(first: Int): [User!]!
  createdAt: DateTime!
}

type Profile {
  bio: String
  avatar: String
  website: String
  location: String
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  likes: Int!
  createdAt: DateTime!
  updatedAt: DateTime!
}

Should I add pagination to followers/following? Is the nesting too deep?""",
            "SKIP_TECHNICAL",
            "realistic_graphql_schema"
        ),
        
        # Assembly/low-level debugging
        (
            """Disassembled binary showing segfault location:

0x00401234 <+0>:     push   %rbp
0x00401235 <+1>:     mov    %rsp,%rbp
0x00401238 <+4>:     sub    $0x10,%rsp
0x0040123c <+8>:     mov    %rdi,-0x8(%rbp)
0x00401240 <+12>:    mov    -0x8(%rbp),%rax
0x00401244 <+16>:    mov    (%rax),%eax        <- SIGSEGV here
0x00401246 <+18>:    add    $0x10,%rsp
0x0040124a <+22>:    pop    %rbp
0x0040124b <+23>:    retq

Register dump:
rax=0x0000000000000000 rbx=0x00007fff5fbff6a0 rcx=0x0000000000000001
rdx=0x0000000000400560 rsi=0x00007fff5fbff6a8 rdi=0x0000000000000000

RAX is null. Dereferencing null pointer at offset +16?""",
            "SKIP_TECHNICAL",
            "realistic_assembly_debug"
        ),
        
        # Makefile compilation error
        (
            """My C++ project won't compile:

make[1]: Entering directory '/home/user/project/src'
g++ -c -Wall -Wextra -O2 -std=c++17 main.cpp -o main.o
g++ -c -Wall -Wextra -O2 -std=c++17 utils.cpp -o utils.o
utils.cpp: In function 'std::vector<int> parseNumbers(std::string)':
utils.cpp:45:23: error: 'stoi' is not a member of 'std'
   45 |         numbers.push_back(std::stoi(token));
      |                           ^~~~~
make[1]: *** [Makefile:12: utils.o] Error 1

I'm including <string>. Why isn't stoi found?""",
            "SKIP_TECHNICAL",
            "realistic_makefile_compile"
        ),
        
        # Android Logcat output
        (
            """App crashing on Android. Logcat shows:

E/AndroidRuntime: FATAL EXCEPTION: main
E/AndroidRuntime: Process: com.example.myapp, PID: 12345
E/AndroidRuntime: java.lang.RuntimeException: Unable to start activity ComponentInfo
E/AndroidRuntime: Caused by: android.view.InflateException: Binary XML file line #27
E/AndroidRuntime: Caused by: java.lang.ClassNotFoundException: Didn't find class "androidx.constraintlayout.widget.ConstraintLayout"
W/System  : ClassLoader referenced unknown path: /data/app/com.example.myapp-1/lib/arm64
I/Process : Sending signal. PID: 12345 SIG: 9

Missing ConstraintLayout dependency?""",
            "SKIP_TECHNICAL",
            "realistic_android_logcat"
        ),
        
        # Nginx configuration
        (
            """My nginx reverse proxy isn't working:

server {
    listen 80;
    server_name example.com www.example.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api {
        proxy_pass http://localhost:8080;
    }
}

Getting 502 Bad Gateway. Backend servers are running. What's wrong?""",
            "SKIP_TECHNICAL",
            "realistic_nginx_config"
        ),
        
        # Rust compiler errors
        (
            r"""Rust won't compile my code:

error[E0382]: borrow of moved value: `data`
  --> src/main.rs:45:23
   |
42 |     let data = vec![1, 2, 3, 4, 5];
   |         ---- move occurs because `data` has type `Vec<i32>`, which does not implement the `Copy` trait
43 |     process_data(data);
   |                  ---- value moved here
44 |     
45 |     for item in data.iter() {
   |                 ^^^^ value borrowed here after move

error[E0597]: `temp` does not live long enough
  --> src/main.rs:67:19
   |
66 |     let temp = String::from("hello");
67 |     let reference = &temp;
   |                     ^^^^^ borrowed value does not live long enough
70 | }
   | - `temp` dropped here while still borrowed

I don't understand Rust's ownership system yet. Help?""",
            "SKIP_TECHNICAL",
            "realistic_rust_borrow_checker"
        ),
        
        # Apache server log analysis
        (
            """Suspicious activity in Apache access log:

192.168.1.100 - - [05/Oct/2024:14:23:45 +0000] "GET /admin/login.php HTTP/1.1" 404 512
192.168.1.100 - - [05/Oct/2024:14:23:46 +0000] "POST /wp-login.php HTTP/1.1" 404 512
192.168.1.100 - - [05/Oct/2024:14:23:47 +0000] "GET /../../../etc/passwd HTTP/1.1" 400 345
192.168.1.100 - - [05/Oct/2024:14:23:48 +0000] "GET /phpMyAdmin/ HTTP/1.1" 404 512
45.76.123.234 - - [05/Oct/2024:14:24:01 +0000] "GET / HTTP/1.1" 200 4523
45.76.123.234 - - [05/Oct/2024:14:24:02 +0000] "POST /contact HTTP/1.1" 200 1234

Is this a bot scanning for vulnerabilities? Should I block that IP?""",
            "SKIP_TECHNICAL",
            "realistic_apache_log_analysis"
        ),
        
        # Bash script debugging
        (
            r"""My backup script is failing:

#!/bin/bash
set -e

BACKUP_DIR="/var/backups/db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATABASE="myapp_production"

echo "Starting backup at $TIMESTAMP"

pg_dump -U postgres -h localhost $DATABASE > $BACKUP_DIR/backup_$TIMESTAMP.sql

if [ $? -eq 0 ]; then
    echo "Backup successful"
    gzip $BACKUP_DIR/backup_$TIMESTAMP.sql
    
    # Delete backups older than 7 days
    find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
else
    echo "Backup failed!"
    exit 1
fi

Error: /var/backups/db/backup_20241005_142345.sql: No such file or directory

The directory exists and has correct permissions. What's happening?""",
            "SKIP_TECHNICAL",
            "realistic_bash_script_debug"
        ),
        
        # PowerShell script error
        (
            """PowerShell script throwing errors:

$servers = @("server1", "server2", "server3")
$results = @()

foreach ($server in $servers) {
    try {
        $response = Invoke-WebRequest -Uri "http://$server/health" -TimeoutSec 5
        $results += [PSCustomObject]@{
            Server = $server
            Status = "Online"
            ResponseTime = $response.ResponseTime
        }
    }
    catch {
        $results += [PSCustomObject]@{
            Server = $server
            Status = "Offline"
            Error = $_.Exception.Message
        }
    }
}

$results | Export-Csv -Path "C:\\health-check.csv" -NoTypeInformation

Error: Exception calling "Invoke-WebRequest" with "1" argument(s): "Unable to connect to the remote server"

All servers are reachable via ping. Why?""",
            "SKIP_TECHNICAL",
            "realistic_powershell_error"
        ),
        
        # Elasticsearch query DSL
        (
            """My Elasticsearch query is too slow:

GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "description": "laptop"
          }
        },
        {
          "range": {
            "price": {
              "gte": 500,
              "lte": 2000
            }
          }
        }
      ],
      "should": [
        {
          "term": {
            "brand": "dell"
          }
        },
        {
          "term": {
            "brand": "hp"
          }
        }
      ],
      "filter": [
        {
          "term": {
            "in_stock": true
          }
        }
      ]
    }
  },
  "aggs": {
    "brands": {
      "terms": {
        "field": "brand.keyword",
        "size": 20
      }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 500 },
          { "from": 500, "to": 1000 },
          { "from": 1000 }
        ]
      }
    }
  }
}

Takes 8 seconds on 10M documents. Need indexing advice?""",
            "SKIP_TECHNICAL",
            "realistic_elasticsearch_query"
        ),
        
        # Redis CLI session
        (
            """Redis showing high memory usage:

redis-cli> INFO memory
used_memory:15728640000
used_memory_human:14.64G
used_memory_peak:16777216000
used_memory_peak_human:15.62G

redis-cli> DBSIZE
(integer) 12456789

redis-cli> KEYS user:session:*
1) "user:session:abc123"
2) "user:session:def456"
... (showing 2 of 8500000 keys)

redis-cli> TTL user:session:abc123
(integer) -1

Most session keys have no expiration! How do I set TTL on existing keys?""",
            "SKIP_TECHNICAL",
            "realistic_redis_memory"
        ),
        
        # MongoDB aggregation pipeline
        (
            """MongoDB aggregation running out of memory:

db.orders.aggregate([
  {
    $match: {
      created_at: { $gte: ISODate("2024-01-01") },
      status: "completed"
    }
  },
  {
    $lookup: {
      from: "users",
      localField: "user_id",
      foreignField: "_id",
      as: "user"
    }
  },
  {
    $unwind: "$user"
  },
  {
    $lookup: {
      from: "products",
      localField: "items.product_id",
      foreignField: "_id",
      as: "products"
    }
  },
  {
    $group: {
      _id: "$user.email",
      total_spent: { $sum: "$total_amount" },
      order_count: { $sum: 1 },
      products: { $push: "$products" }
    }
  },
  {
    $sort: { total_spent: -1 }
  },
  {
    $limit: 100
  }
])

Error: Exceeded memory limit for $group. Use allowDiskUse: true?""",
            "SKIP_TECHNICAL",
            "realistic_mongodb_aggregation"
        ),
        
        # Jenkins pipeline syntax
        (
            """Jenkins pipeline failing at deploy stage:

pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_NAME = 'myapp'
        KUBECONFIG = credentials('k8s-config')
    }
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} .'
            }
        }
        
        stage('Test') {
            steps {
                sh 'docker run ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} npm test'
            }
        }
        
        stage('Push') {
            steps {
                sh 'docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}'
            }
        }
        
        stage('Deploy') {
            steps {
                sh '''
                    kubectl set image deployment/myapp \
                        myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} \
                        --record
                    kubectl rollout status deployment/myapp
                '''
            }
        }
    }
}

Error: line 2: kubectl: command not found. kubectl is installed. PATH issue?""",
            "SKIP_TECHNICAL",
            "realistic_jenkins_pipeline"
        ),
        
        # CircleCI configuration
        (
            """CircleCI workflow not running tests:

version: 2.1

executors:
  node-executor:
    docker:
      - image: cimg/node:18.17
    working_directory: ~/project

jobs:
  build:
    executor: node-executor
    steps:
      - checkout
      - restore_cache:
          keys:
            - v1-dependencies-{{ checksum "package-lock.json" }}
            - v1-dependencies-
      - run:
          name: Install Dependencies
          command: npm ci
      - save_cache:
          paths:
            - node_modules
          key: v1-dependencies-{{ checksum "package-lock.json" }}
      - run:
          name: Run Tests
          command: npm test
      - store_test_results:
          path: test-results
      - store_artifacts:
          path: coverage

workflows:
  version: 2
  build_and_test:
    jobs:
      - build:
          filters:
            branches:
              only:
                - main
                - develop

Tests run locally but fail on CircleCI with "Cannot find module '../src/config'". Why?""",
            "SKIP_TECHNICAL",
            "realistic_circleci_config"
        ),
        
        # Ansible playbook error
        (
            """Ansible playbook failing on remote host:

---
- name: Deploy web application
  hosts: webservers
  become: yes
  
  vars:
    app_user: webapp
    app_dir: /var/www/myapp
    
  tasks:
    - name: Create application user
      user:
        name: "{{ app_user }}"
        system: yes
        shell: /bin/bash
        
    - name: Create application directory
      file:
        path: "{{ app_dir }}"
        state: directory
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        mode: '0755'
        
    - name: Copy application files
      synchronize:
        src: ../dist/
        dest: "{{ app_dir }}"
        delete: yes
        
    - name: Install PM2
      npm:
        name: pm2
        global: yes
        
    - name: Start application
      shell: |
        cd {{ app_dir }}
        pm2 start app.js --name myapp
      become_user: "{{ app_user }}"

TASK [Start application] ****
fatal: [web-01]: FAILED! => {"msg": "cd /var/www/myapp\\npm2 start app.js --name myapp: command not found"}

Newline issue in the shell command?""",
            "SKIP_TECHNICAL",
            "realistic_ansible_playbook"
        ),
        
        # Wireshark packet capture analysis
        (
            """Wireshark capture showing network issues:

Frame 1234: 66 bytes on wire, 66 bytes captured
Ethernet II, Src: 00:1a:2b:3c:4d:5e, Dst: ff:ff:ff:ff:ff:ff
Internet Protocol Version 4, Src: 192.168.1.100, Dst: 192.168.1.1
Transmission Control Protocol, Src Port: 52341, Dst Port: 443, Seq: 1, Ack: 1, Len: 0
    [TCP Retransmission]
    
Frame 1235: 66 bytes on wire, 66 bytes captured  
Ethernet II, Src: 00:1a:2b:3c:4d:5e, Dst: ff:ff:ff:ff:ff:ff
Internet Protocol Version 4, Src: 192.168.1.100, Dst: 192.168.1.1
Transmission Control Protocol, Src Port: 52341, Dst Port: 443, Seq: 1, Ack: 1, Len: 0
    [TCP Retransmission]

Frame 1236: 66 bytes on wire, 66 bytes captured
[TCP Retransmission]

Seeing lots of retransmissions. MTU problem? Firewall dropping packets?""",
            "SKIP_TECHNICAL",
            "realistic_wireshark_capture"
        ),
        
        # Postman collection/API testing
        (
            """Postman tests failing for API endpoint:

POST https://api.example.com/v1/users
Headers:
  Content-Type: application/json
  Authorization: Bearer eyJhbGc...
Body:
{
  "email": "test@example.com",
  "password": "SecurePass123!",
  "name": "Test User"
}

Response: 422 Unprocessable Entity
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "password",
        "message": "Password must contain at least one lowercase letter"
      }
    ]
  }
}

But my password DOES have lowercase letters! Is the regex wrong on their end?""",
            "SKIP_TECHNICAL",
            "realistic_postman_api_test"
        ),
        
        # Selenium WebDriver error
        (
            """Selenium test failing intermittently:

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("https://example.com/login")

username = driver.find_element(By.ID, "username")
password = driver.find_element(By.ID, "password")
submit = driver.find_element(By.ID, "submit-btn")

username.send_keys("testuser")
password.send_keys("password123")
submit.click()

WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "dashboard"))
)

Error: selenium.common.exceptions.TimeoutException: Message: 
No element found after 10 seconds

Works locally but fails in CI. Timing issue?""",
            "SKIP_TECHNICAL",
            "realistic_selenium_test"
        ),
        
        # Pytest fixtures and markers
        (
            """Pytest tests not using fixtures correctly:

import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_database():
    db = Mock()
    db.query.return_value = [{"id": 1, "name": "Test"}]
    return db

@pytest.fixture
def mock_api_client():
    client = Mock()
    client.get.return_value = {"status": "success"}
    return client

@pytest.mark.asyncio
async def test_user_service(mock_database, mock_api_client):
    service = UserService(mock_database, mock_api_client)
    result = await service.get_user(1)
    assert result["name"] == "Test"
    mock_database.query.assert_called_once()

ERROR: fixture 'mock_database' not found
E       fixture 'mock_api_client' not found

Fixtures are defined in same file. Why can't pytest find them?""",
            "SKIP_TECHNICAL",
            "realistic_pytest_fixtures"
        ),
        
        # JWT token debugging
        (
            """JWT validation failing. Token details:

Header:
{
  "alg": "HS256",
  "typ": "JWT"
}

Payload:
{
  "sub": "1234567890",
  "name": "John Doe",
  "email": "john@example.com",
  "iat": 1728134623,
  "exp": 1728138223,
  "iss": "https://auth.example.com",
  "aud": "https://api.example.com"
}

Signature:
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret_key_12345
)

Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiZW1haWwiOiJqb2huQGV4YW1wbGUuY29tIiwiaWF0IjoxNzI4MTM0NjIzLCJleHAiOjE3MjgxMzgyMjMsImlzcyI6Imh0dHBzOi8vYXV0aC5leGFtcGxlLmNvbSIsImF1ZCI6Imh0dHBzOi8vYXBpLmV4YW1wbGUuY29tIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Validation error: "Invalid signature". But I'm using the same secret! What's wrong?""",
            "SKIP_TECHNICAL",
            "realistic_jwt_debugging"
        ),
        
        # Load testing results with Locust/JMeter
        (
            """Load test results showing performance degradation:

Summary Report:
Total Requests: 100,000
Duration: 600 seconds (10 minutes)
Success Rate: 94.5%

Response Time Percentiles:
50th percentile: 245ms
75th percentile: 567ms
90th percentile: 1,234ms
95th percentile: 2,456ms
99th percentile: 8,901ms

Errors:
- 500 Internal Server Error: 3,200 (3.2%)
- 504 Gateway Timeout: 2,300 (2.3%)

RPS (Requests Per Second):
0-2min: 180 RPS (stable)
2-5min: 165 RPS (slight drop)
5-8min: 142 RPS (degrading)
8-10min: 98 RPS (severe degradation)

Database connections: Maxed out at 50/50
Memory usage: Climbed from 2GB to 14GB
CPU: Steady at 85%

Memory leak? Connection pool exhaustion? Need help diagnosing.""",
            "SKIP_TECHNICAL",
            "realistic_load_test_results"
        ),
        
        # ADDITIONAL COMPLEX MULTI-LINE FAST-PATH TESTS
        
        # Complex Terraform Configuration
        (
            """I'm setting up infrastructure as code with Terraform. Here's my main.tf file:

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "main-vpc"
    Environment = "production"
  }
}

resource "aws_subnet" "public" {
  count = 2
  vpc_id = aws_vpc.main.id
  cidr_block = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "public-subnet-${count.index + 1}"
  }
}

resource "aws_security_group" "web" {
  name_prefix = "web-sg-"
  vpc_id = aws_vpc.main.id
  
  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

Should I add variables for the CIDR blocks and region? How do I handle state locking?""",
            "SKIP_TECHNICAL",
            "complex_terraform_multiline"
        ),
        
        # Complex GitHub Actions Workflow
        (
            """My CI/CD pipeline is failing and I need help debugging this GitHub Actions workflow:

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests with coverage
      run: |
        pytest --cov=myapp --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: myapp
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to ECS
      uses: aws-actions/amazon-ecs-deploy-task-definition@v1
      with:
        task-definition: task-definition.json
        service: myapp-service
        cluster: production-cluster
        wait-for-service-stability: true

The test job passes but the deploy job fails with "The security token included in the request is invalid". What's wrong?""",
            "SKIP_TECHNICAL",
            "complex_github_actions_multiline"
        ),
        
        # Complex Regular Expressions
        (
            r"""I'm trying to parse complex log files with this regex pattern:

^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(?P<thread>[^\]]+)\] (?P<level>(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)) (?P<logger>[^\s]+) - (?P<message>(?:(?! - ).)*)(?: - (?P<exception>.*))?$

It should match log entries like:
2023-10-05 14:23:45,123 [http-nio-8080-exec-1] INFO com.example.MyClass - Processing user request for id: 12345
2023-10-05 14:23:46,456 [http-nio-8080-exec-2] ERROR com.example.MyClass - Database connection failed - java.sql.SQLException: Connection timeout
2023-10-05 14:23:47,789 [http-nio-8080-exec-3] DEBUG com.example.MyClass - Cache hit for key: user:12345

But it's not capturing the exception details properly. The issue is with the negative lookahead. Should I use a different approach for multi-line exceptions?

Also, how do I handle stack traces that span multiple lines after the initial log message?""",
            "SKIP_TECHNICAL",
            "complex_regex_multiline"
        ),
        
        # Complex CSS with Media Queries
        (
            """My responsive design is broken. Here's the CSS I'm using:

:root {
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --font-size-base: 16px;
  --spacing-unit: 1rem;
  --border-radius: 4px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--spacing-unit);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: calc(var(--spacing-unit) * 2);
}

.card {
  background: white;
  border-radius: var(--border-radius);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: var(--spacing-unit);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

@media (max-width: 768px) {
  .container {
    padding: 0 calc(var(--spacing-unit) / 2);
  }
  
  .grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-unit);
  }
  
  .card {
    padding: calc(var(--spacing-unit) / 2);
  }
}

@media (max-width: 480px) {
  :root {
    --font-size-base: 14px;
    --spacing-unit: 0.75rem;
  }
  
  .card {
    border-radius: calc(var(--border-radius) / 2);
  }
}

@media (prefers-reduced-motion: reduce) {
  .card {
    transition: none;
  }
  
  .card:hover {
    transform: none;
  }
}

The grid isn't stacking properly on mobile. What am I doing wrong with the media queries?""",
            "SKIP_TECHNICAL",
            "complex_css_media_queries_multiline"
        ),
        
        # Complex SQL with Window Functions
        (
            """This analytical query is taking too long. Can you help optimize it?

WITH monthly_stats AS (
  SELECT 
    DATE_TRUNC('month', created_at) as month,
    user_id,
    COUNT(*) as orders_count,
    SUM(total_amount) as total_spent,
    AVG(total_amount) as avg_order_value,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) as order_rank
  FROM orders 
  WHERE created_at >= '2023-01-01' 
    AND status = 'completed'
  GROUP BY DATE_TRUNC('month', created_at), user_id
),
user_segments AS (
  SELECT 
    user_id,
    CASE 
      WHEN total_spent > 1000 THEN 'high_value'
      WHEN total_spent > 500 THEN 'medium_value' 
      ELSE 'low_value'
    END as segment,
    total_spent,
    orders_count,
    avg_order_value
  FROM (
    SELECT 
      user_id,
      SUM(total_spent) as total_spent,
      SUM(orders_count) as orders_count,
      AVG(avg_order_value) as avg_order_value
    FROM monthly_stats
    GROUP BY user_id
  ) user_totals
),
cohort_analysis AS (
  SELECT 
    DATE_TRUNC('month', u.created_at) as cohort_month,
    DATE_TRUNC('month', o.created_at) as order_month,
    COUNT(DISTINCT CASE WHEN o.created_at IS NOT NULL THEN u.id END) as active_users,
    COUNT(DISTINCT u.id) as total_users
  FROM users u
  LEFT JOIN orders o ON u.id = o.user_id 
    AND o.status = 'completed'
    AND o.created_at >= u.created_at
  WHERE u.created_at >= '2023-01-01'
  GROUP BY DATE_TRUNC('month', u.created_at), DATE_TRUNC('month', o.created_at)
)
SELECT 
  cs.cohort_month,
  cs.order_month,
  cs.active_users,
  cs.total_users,
  ROUND(cs.active_users::decimal / cs.total_users, 4) as retention_rate,
  us.segment,
  us.total_spent,
  us.orders_count,
  us.avg_order_value
FROM cohort_analysis cs
CROSS JOIN user_segments us
ORDER BY cs.cohort_month, cs.order_month, us.total_spent DESC;

The query runs for 45 minutes on our production database. Should I add indexes? Use materialized views? Break it into smaller queries?""",
            "SKIP_TECHNICAL",
            "complex_sql_window_functions_multiline"
        ),
        
        # Complex JavaScript Async/Await with Error Handling
        (
            """My async function is not handling errors properly. Here's the code:

const axios = require('axios');
const fs = require('fs').promises;

class DataProcessor {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    this.retryAttempts = 3;
  }
  
  async fetchData(endpoint) {
    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await axios.get(`${this.apiUrl}${endpoint}`);
        return response.data;
      } catch (error) {
        if (attempt === this.retryAttempts) {
          throw new Error(`Failed after ${this.retryAttempts} attempts: ${error.message}`);
        }
        await this.delay(1000 * attempt);
      }
    }
  }
  
  async processBatch(items) {
    const results = [];
    for (const item of items) {
      try {
        const processed = await this.processItem(item);
        results.push(processed);
      } catch (error) {
        console.error(`Failed to process item ${item.id}: ${error.message}`);
      }
    }
    return results;
  }
  
  async processItem(item) {
    const additionalData = await this.fetchData(`/items/${item.id}/details`);
    const transformed = {
      ...item,
      ...additionalData,
      processed_at: new Date().toISOString()
    };
    await fs.writeFile(`${item.id}.json`, JSON.stringify(transformed, null, 2));
    return transformed;
  }
  
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

The issue is that errors in processItem are not being caught properly by processBatch. What's wrong with my error handling?""",
            "SKIP_TECHNICAL",
            "complex_javascript_async_multiline"
        ),
        
        # NEW TESTS THAT SHOULD BE SKIPPED - 19 additional technical scenarios
        
        # Complex CSS with multiple selectors
        (
            """.container { display: flex; } .item:nth-child(2n) { background: #f0f0f0; } @media (max-width: 768px) { .container { flex-direction: column; } } .item::before { content: ""; position: absolute; } .header > nav > ul > li > a:hover { color: blue; }""",
            "SKIP_TECHNICAL",
            "complex_css_selectors"
        ),
        
        # GraphQL schema definition
        (
            """type User { id: ID! name: String! email: String! posts: [Post!]! } type Post { id: ID! title: String! content: String! author: User! comments: [Comment!]! } type Comment { id: ID! text: String! author: User! }""",
            "SKIP_TECHNICAL",
            "graphql_schema_definition"
        ),
        
        # Terraform configuration
        (
            """resource "aws_instance" "web" { ami = "ami-12345678" instance_type = "t2.micro" tags = { Name = "WebServer" Environment = "Production" } } resource "aws_s3_bucket" "data" { bucket = "my-data-bucket" acl = "private" }""",
            "SKIP_TECHNICAL",
            "terraform_configuration"
        ),
        
        # Nginx configuration
        (
            """server { listen 80; server_name example.com; location / { proxy_pass http://localhost:3000; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; } location /static/ { alias /var/www/static/; expires 30d; } }""",
            "SKIP_TECHNICAL",
            "nginx_server_config"
        ),
        
        # CSV data with many columns
        (
            """id,name,email,age,city,country,phone,status,created_at,updated_at
1,John,john@example.com,30,NYC,USA,555-0001,active,2024-01-01,2024-01-15
2,Jane,jane@example.com,25,LA,USA,555-0002,inactive,2024-01-02,2024-01-16
3,Bob,bob@example.com,35,Chicago,USA,555-0003,active,2024-01-03,2024-01-17
4,Alice,alice@example.com,28,Boston,USA,555-0004,active,2024-01-04,2024-01-18
5,Charlie,charlie@example.com,32,Seattle,USA,555-0005,inactive,2024-01-05,2024-01-19""",
            "SKIP_TECHNICAL",
            "csv_data_table"
        ),
        
        # Multiple environment files
        (
            """Production: DATABASE_URL=prod.db.com API_KEY=pk_live_xyz PORT=8080
Staging: DATABASE_URL=staging.db.com API_KEY=pk_test_abc PORT=8081  
Development: DATABASE_URL=localhost:5432 API_KEY=pk_dev_123 PORT=3000""",
            "SKIP_TECHNICAL",
            "multiple_env_configs"
        ),
        
        # Makefile commands
        (
            """build: gcc -o app main.c utils.c -Wall -O2
test: ./app --test-mode
clean: rm -f *.o app
install: cp app /usr/local/bin/
run: ./app --verbose
docker: docker build -t myapp . && docker run myapp""",
            "SKIP_TECHNICAL",
            "makefile_targets"
        ),
        
        # SSH config file
        (
            """Host production
    HostName prod.example.com
    User deploy
    Port 22
    IdentityFile ~/.ssh/prod_key
Host staging
    HostName staging.example.com
    User admin
    Port 2222
    IdentityFile ~/.ssh/staging_key""",
            "SKIP_TECHNICAL",
            "ssh_config_multiple_hosts"
        ),
        
        # Cron job schedule
        (
            """0 0 * * * /usr/local/bin/backup.sh
*/5 * * * * /usr/local/bin/health_check.sh
0 2 * * 0 /usr/local/bin/weekly_report.sh
0 */4 * * * /usr/local/bin/cleanup.sh
30 3 1 * * /usr/local/bin/monthly_task.sh""",
            "SKIP_TECHNICAL",
            "cron_job_schedules"
        ),
        
        # DNS zone file
        (
            """example.com. IN A 192.0.2.1
www IN CNAME example.com.
mail IN A 192.0.2.2
@ IN MX 10 mail.example.com.
@ IN TXT "v=spf1 mx ~all"
_dmarc IN TXT "v=DMARC1; p=none;"
ftp IN A 192.0.2.3""",
            "SKIP_TECHNICAL",
            "dns_zone_records"
        ),
        
        # Binary/hex dump
        (
            """0000: 48 65 6c 6c 6f 20 57 6f 72 6c 64 21 0a 54 68 69
0010: 73 20 69 73 20 61 20 74 65 73 74 20 66 69 6c 65
0020: 2e 20 49 74 20 63 6f 6e 74 61 69 6e 73 20 62 69
0030: 6e 61 72 79 20 64 61 74 61 2e 00 00 00 00 00 00""",
            "SKIP_TECHNICAL",
            "hex_dump_binary"
        ),
        
        # Apache htaccess
        (
            r"""RewriteEngine On
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
<FilesMatch "\.(jpg|jpeg|png|gif)$">
    Header set Cache-Control "max-age=604800, public"
</FilesMatch>
ErrorDocument 404 /404.html
Options -Indexes""",
            "SKIP_TECHNICAL",
            "apache_htaccess_rules"
        ),
        
        # Git log output
        (
            """commit a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
Author: John Doe <john@example.com>
Date: Mon Oct 6 10:30:00 2025 -0700
    Fix authentication bug in login controller
commit b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0a1
Author: Jane Smith <jane@example.com>
Date: Sun Oct 5 15:20:00 2025 -0700
    Add user profile page component""",
            "SKIP_TECHNICAL",
            "git_log_commits"
        ),
        
        # Kubernetes manifest
        (
            """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80""",
            "SKIP_TECHNICAL",
            "kubernetes_deployment_manifest"
        ),
        
        # AWS CloudFormation template
        (
            """Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: my-app-bucket
      VersioningConfiguration:
        Status: Enabled
  MyInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: t2.micro
      KeyName: mykey""",
            "SKIP_TECHNICAL",
            "cloudformation_template"
        ),
        
        # System log excerpt
        (
            """[2025-10-06 10:30:15] INFO: Application started on port 8080
[2025-10-06 10:30:16] DEBUG: Connected to database at postgresql://localhost:5432/mydb
[2025-10-06 10:30:20] WARN: Rate limit exceeded for IP 192.168.1.100
[2025-10-06 10:30:25] ERROR: Failed to process request: Connection timeout
[2025-10-06 10:30:30] FATAL: Out of memory error, shutting down""",
            "SKIP_TECHNICAL",
            "system_log_entries"
        ),
        
        # Network packet trace
        (
            """10:30:15.123456 IP 192.168.1.100.54321 > 192.168.1.1.80: Flags [S], seq 1234567890, win 65535
10:30:15.123567 IP 192.168.1.1.80 > 192.168.1.100.54321: Flags [S.], seq 9876543210, ack 1234567891, win 65535
10:30:15.123678 IP 192.168.1.100.54321 > 192.168.1.1.80: Flags [.], ack 1, win 65535""",
            "SKIP_TECHNICAL",
            "network_packet_trace"
        ),
        
        # Assembly code snippet
        (
            """mov eax, [ebp+8]
add eax, [ebp+12]
mov [ebp-4], eax
push eax
call printf
add esp, 4
mov eax, [ebp-4]
pop ebp
ret""",
            "SKIP_TECHNICAL",
            "assembly_code_x86"
        ),
        
        # Regular expressions list
        (
            r"""Email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
Phone: /^\+?1?\d{9,15}$/
URL: /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/
IPv4: /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/
Date: /^\d{4}-\d{2}-\d{2}$/""",
            "SKIP_TECHNICAL",
            "regex_patterns_list"
        ),
    ]
    
    # Layer 3: Semantic Classification Tests
    SEMANTIC_TESTS = [
        # Technical Content - Code & Programming
        (
            "Here's my Python function that calculates Fibonacci numbers using dynamic programming with memoization",
            "SKIP_TECHNICAL",
            "technical_code_python"
        ),
        (
            "The algorithm uses a binary search tree with O(log n) time complexity for insertions",
            "SKIP_TECHNICAL",
            "technical_code_algorithm"
        ),
        (
            "This React component manages state using hooks and renders a list of items",
            "SKIP_TECHNICAL",
            "technical_code_react"
        ),
        (
            "The SQL query joins three tables using LEFT JOIN and filters by date range",
            "SKIP_TECHNICAL",
            "technical_code_sql"
        ),
        (
            "Implement a singleton pattern with thread-safe lazy initialization",
            "SKIP_TECHNICAL",
            "technical_code_pattern"
        ),
        
        # Technical Content - Error Messages
        (
            "ERROR: Exception in thread main java.lang.NullPointerException at line 42 in MyClass.java",
            "SKIP_TECHNICAL",
            "technical_error_java"
        ),
        (
            "Traceback (most recent call last): File app.py line 15 in main IndexError: list index out of range",
            "SKIP_TECHNICAL",
            "technical_error_python"
        ),
        (
            "Error 404: Resource not found at /api/v1/users/123",
            "SKIP_TECHNICAL",
            "technical_error_http"
        ),
        (
            "Segmentation fault core dumped at memory address 0x7fff5fbff710",
            "SKIP_TECHNICAL",
            "technical_error_segfault"
        ),
        
        # Technical Content - API & Documentation
        (
            "The REST API endpoint accepts POST requests with JSON payload containing user credentials",
            "SKIP_TECHNICAL",
            "technical_api_rest"
        ),
        (
            "GraphQL mutation updateUser requires id email and optionally accepts name and avatar fields",
            "SKIP_TECHNICAL",
            "technical_api_graphql"
        ),
        (
            "The configuration file uses YAML format with nested properties for database connection settings",
            "SKIP_TECHNICAL",
            "technical_api_config"
        ),
        (
            "WebSocket connection established on port 8080 with binary message protocol",
            "SKIP_TECHNICAL",
            "technical_api_websocket"
        ),
        
        # Technical Content - System & Logs
        (
            "INFO 2025-10-05 14:23:45 Server started on port 3000",
            "SKIP_TECHNICAL",
            "technical_log_info"
        ),
        (
            "WARN Database connection pool exhausted retrying in 5 seconds",
            "SKIP_TECHNICAL",
            "technical_log_warn"
        ),
        (
            "DEBUG Request headers: Content-Type application/json Authorization Bearer token",
            "SKIP_TECHNICAL",
            "technical_log_debug"
        ),
        
        # Technical Content - Abstract Questions (moved to FACTUAL_QUERY - these ask ABOUT concepts, not actual implementations)
        
        
        # Output Formatting - Format Instructions
        (
            "Please format your response as a JSON object with keys and values",
            "SKIP_INSTRUCTION",
            "format_instruction_json"
        ),
        (
            "Return the data in CSV format with headers",
            "SKIP_INSTRUCTION",
            "format_instruction_csv"
        ),
        (
            "Structure the output as YAML with nested properties",
            "SKIP_INSTRUCTION",
            "format_instruction_yaml"
        ),
        (
            "Put the results in a markdown table",
            "SKIP_INSTRUCTION",
            "format_instruction_table"
        ),
        (
            "Format as a numbered list with subsections",
            "SKIP_INSTRUCTION",
            "format_instruction_list"
        ),
        
        # Output Formatting - Style Adjustments
        (
            "Can you make that shorter and use bullet points instead?",
            "SKIP_INSTRUCTION",
            "format_style_shorter"
        ),
        (
            "Make it more detailed and comprehensive",
            "SKIP_INSTRUCTION",
            "format_style_longer"
        ),
        (
            "Simplify the explanation for a beginner",
            "SKIP_INSTRUCTION",
            "format_style_simple"
        ),
        (
            "Provide a more technical detailed explanation",
            "SKIP_INSTRUCTION",
            "format_style_technical"
        ),
        
        # Output Formatting - Tone Changes
        (
            "Rewrite the previous answer in a more formal tone",
            "SKIP_INSTRUCTION",
            "format_tone_formal"
        ),
        (
            "Make it more casual and conversational",
            "SKIP_INSTRUCTION",
            "format_tone_casual"
        ),
        (
            "Write it in a professional business style",
            "SKIP_INSTRUCTION",
            "format_tone_professional"
        ),
        (
            "Explain like I'm five years old",
            "SKIP_INSTRUCTION",
            "format_tone_eli5"
        ),
        
        # Output Formatting - Rewrites
        (
            "Rephrase that in different words",
            "SKIP_INSTRUCTION",
            "format_rewrite_rephrase"
        ),
        (
            "Summarize the previous response",
            "SKIP_INSTRUCTION",
            "format_rewrite_summarize"
        ),
        (
            "Translate the output to Spanish",
            "SKIP_INSTRUCTION",
            "format_rewrite_translate"
        ),
        
        # Pure Math - Arithmetic
        (
            "What is 15 percent of 250?",
            "SKIP_PURE_MATH",
            "math_percentage_basic"
        ),
        (
            "Calculate 45 times 67 equals what?",
            "SKIP_PURE_MATH",
            "math_arithmetic_multiply"
        ),
        (
            "What is 1234 plus 5678 minus 890?",
            "SKIP_PURE_MATH",
            "math_arithmetic_mixed"
        ),
        (
            "Divide 9876 by 12",
            "SKIP_PURE_MATH",
            "math_arithmetic_divide"
        ),
        (
            "What's 23 squared?",
            "SKIP_PURE_MATH",
            "math_arithmetic_square"
        ),
        
        # Pure Math - Conversions
        (
            "Convert 100 kilometers to miles",
            "SKIP_PURE_MATH",
            "math_conversion_distance"
        ),
        (
            "72 fahrenheit to celsius",
            "SKIP_PURE_MATH",
            "math_conversion_temperature"
        ),
        (
            "How many ounces in 2.5 pounds?",
            "SKIP_PURE_MATH",
            "math_conversion_weight"
        ),
        (
            "Convert 500 milliliters to cups",
            "SKIP_PURE_MATH",
            "math_conversion_volume"
        ),
        
        # Pure Math - Equations
        (
            "Solve for x in equation 2x plus 5 equals 15",
            "SKIP_PURE_MATH",
            "math_equation_linear"
        ),
        (
            "What is the square root of 144?",
            "SKIP_PURE_MATH",
            "math_equation_sqrt"
        ),
        (
            "Calculate the area of a circle with radius 5",
            "SKIP_PURE_MATH",
            "math_equation_circle"
        ),
        (
            "Find the volume of a cube with side length 10",
            "SKIP_PURE_MATH",
            "math_equation_volume"
        ),
        
        # Pure Math - Percentages
        (
            "What's 30 percent of 850?",
            "SKIP_PURE_MATH",
            "math_percentage_calc"
        ),
        (
            "Calculate discount: 120 dollars minus 25 percent",
            "SKIP_PURE_MATH",
            "math_percentage_discount"
        ),
        (
            "If 40 is 80 percent of a number, what's the number?",
            "SKIP_PURE_MATH",
            "math_percentage_reverse"
        ),
        
        # Translation - Explicit with Text
        (
            "Translate this to Spanish: Hello, how are you today?",
            "SKIP_TRANSLATION",
            "translation_explicit_spanish"
        ),
        (
            "How do you say 'good morning' in French?",
            "SKIP_TRANSLATION",
            "translation_phrase_french"
        ),
        (
            "Convert this English text to Japanese: The weather is nice",
            "SKIP_TRANSLATION",
            "translation_explicit_japanese"
        ),
        (
            "What is the German translation of 'I am hungry'?",
            "SKIP_TRANSLATION",
            "translation_phrase_german"
        ),
        (
            "Translate to Italian: Where is the train station?",
            "SKIP_TRANSLATION",
            "translation_explicit_italian"
        ),
        (
            "How to say 'computer' in Russian?",
            "SKIP_TRANSLATION",
            "translation_word_russian"
        ),
        (
            "Translate the following paragraph to Portuguese: This is a test",
            "SKIP_TRANSLATION",
            "translation_explicit_portuguese"
        ),
        
        # Grammar/Proofreading - Fix Grammar
        (
            "Fix the grammar in this text: She don't like going to school",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_fix_basic"
        ),
        (
            "Correct this sentence: Me and him went to the store",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_fix_pronoun"
        ),
        (
            "Check grammar: Their going too the park tomorrow",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_fix_homophones"
        ),
        (
            "Fix: I has three book on my desk",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_fix_agreement"
        ),
        
        # Grammar/Proofreading - Check Typos
        (
            "Check for typos: The quick brown fox jumps over teh lazy dog",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_typo_basic"
        ),
        (
            "Proofread this paragraph: Teh compnay is veyr successfull",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_typo_multiple"
        ),
        (
            "Check spelling in this text: I recieved your mesage yesterday",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_typo_spelling"
        ),
        
        # Grammar/Proofreading - Punctuation
        (
            "Correct the punctuation in this sentence: where are you going today",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_punctuation_missing"
        ),
        (
            "Fix punctuation: I bought apples oranges and bananas",
            "SKIP_GRAMMAR_PROOFREAD",
            "grammar_punctuation_commas"
        ),
        
        # LONG MULTI-SENTENCE SEMANTIC TESTS
        
        # Long Technical Content
        (
            "I've been feeling really overwhelmed at my new job as a software engineer at Microsoft. It's my first role after graduating, and I'm working on a team that handles cloud infrastructure. Everyone seems so experienced and knowledgeable, and I'm constantly worried that I'm not contributing enough or that I'm asking too many basic questions. My mentor has been supportive, but I still feel this constant anxiety about not being good enough. Does this imposter syndrome ever go away? How did you cope with similar feelings when you were starting out in your career?",
            None,
            "long_conversational_career_anxiety"
        ),
        (
            "My daughter Emma is turning 7 next month, and I want to throw her a memorable birthday party. She's really into dinosaurs right now, especially T-Rex and Triceratops. I'm thinking about having it at a local park with maybe 15-20 kids from her class. I'd love ideas for dinosaur-themed games, decorations, and maybe a good cake design. She also has a peanut allergy, so I need to be careful about food options. Do you have any suggestions for making this special for her while keeping it manageable and safe?",
            None,
            "long_conversational_parenting_party"
        ),
        (
            "I'm planning to switch careers from marketing to data science, and I'm feeling both excited and nervous about it. I've always been interested in statistics and analytics, and I've been taking online courses in Python and machine learning for the past six months. I have about 8 years of experience in digital marketing, where I worked a lot with Google Analytics and campaign data. Do you think my marketing background could actually be an advantage in data science? What additional skills should I focus on developing? And how should I approach job hunting when I don't have a traditional data science background?",
            None,
            "long_conversational_career_transition"
        ),
        (
            "My husband and I are thinking about adopting a rescue dog. We've always been dog lovers, but this would be our first pet together. We live in a two-bedroom apartment with a small balcony, and we both work from home three days a week. We're active people who enjoy hiking and running, so we'd love a dog that could join us. However, we're also concerned about adopting an older dog with potential behavioral issues or health problems. What should we look for when visiting shelters? How can we assess if a dog's energy level and personality would be a good fit for our lifestyle? And what questions should we ask the shelter staff?",
            None,
            "long_conversational_pet_adoption"
        ),
        
        # COMPLEX MULTI-LINE TECHNICAL TESTS
        
        # Docker Compose Configuration
        (
            """I'm trying to set up a development environment with Docker Compose. Here's what I need:
            
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret123
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
      
  app:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://admin:secret123@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
    ports:
      - "3000:3000"
    volumes:
      - ./src:/app/src
      
volumes:
  pgdata:

Should I add health checks to ensure services are ready before starting the app? And how do I handle environment variables more securely?""",
            "SKIP_TECHNICAL",
            "complex_docker_compose_multiline"
        ),
        
        # SQL Query with Multiple Joins
        (
            """I need help optimizing this complex SQL query that's running slow:

SELECT 
  u.id,
  u.username,
  u.email,
  COUNT(DISTINCT o.id) as order_count,
  SUM(o.total_amount) as total_spent,
  AVG(o.total_amount) as avg_order_value,
  MAX(o.created_at) as last_order_date,
  COUNT(DISTINCT p.id) as products_purchased,
  ARRAY_AGG(DISTINCT c.name) as categories
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
LEFT JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.id
LEFT JOIN product_categories pc ON p.id = pc.product_id
LEFT JOIN categories c ON pc.category_id = c.id
WHERE u.created_at >= '2024-01-01'
  AND u.status = 'active'
  AND o.status = 'completed'
GROUP BY u.id, u.username, u.email
HAVING COUNT(DISTINCT o.id) > 5
ORDER BY total_spent DESC
LIMIT 100;

The query takes over 30 seconds on our production database. Should I add indexes? Use materialized views? Break it into multiple queries?""",
            "SKIP_TECHNICAL",
            "complex_sql_optimization_multiline"
        ),
        
        # Python Class with Error Handling
        (
            """Here's my implementation of a retry decorator with exponential backoff:

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Max retries reached for {func.__name__}")
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator

Is this implementation thread-safe? Should I add jitter to prevent thundering herd? Any other improvements?""",
            "SKIP_TECHNICAL",
            "complex_python_decorator_multiline"
        ),
        
        # Kubernetes Deployment Configuration
        (
            """I'm deploying my Node.js app to Kubernetes and need review of my configuration:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-nodejs-app
  labels:
    app: nodejs-app
    version: v1.2.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: nodejs-app
  template:
    metadata:
      labels:
        app: nodejs-app
        version: v1.2.0
    spec:
      containers:
      - name: app
        image: myregistry/nodejs-app:1.2.0
        ports:
        - containerPort: 3000
          protocol: TCP
        env:
        - name: NODE_ENV
          value: production
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5

Should I use a Horizontal Pod Autoscaler? Do I need a PodDisruptionBudget for high availability?""",
            "SKIP_TECHNICAL",
            "complex_kubernetes_yaml_multiline"
        ),
        
        # COMPLEX ERROR MESSAGE TESTS
        
        # Stack Trace with Multiple Frames
        (
            """Getting this error when deploying to production:

Traceback (most recent call last):
  File "/app/main.py", line 145, in process_payment
    result = payment_gateway.charge(amount, customer_id)
  File "/app/lib/payment_gateway.py", line 78, in charge
    response = self._make_request('POST', '/charges', data)
  File "/app/lib/payment_gateway.py", line 34, in _make_request
    return self.session.request(method, url, **kwargs)
  File "/usr/local/lib/python3.11/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.11/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
  File "/usr/local/lib/python3.11/site-packages/requests/adapters.py", line 519, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.payment.com', port=443): Max retries exceeded with url: /v1/charges (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f8b9c234d90>: Failed to establish a new connection: [Errno 110] Connection timed out'))

This only happens in production, not staging. Network issue or code problem?""",
            "SKIP_TECHNICAL",
            "complex_python_stacktrace_multiline"
        ),
        
        # JavaScript Console Error
        (
            """My React app is throwing this error in production:

Uncaught Error: Minified React error #31; visit https://reactjs.org/docs/error-decoder.html?invariant=31&args[]=object%20with%20keys%20%7B%7D&args[]= for the full message or use the non-minified dev environment for full errors and additional helpful warnings.
    at ha (chunk-vendors.4f2c8e9b.js:2:25847)
    at hi (chunk-vendors.4f2c8e9b.js:2:67890)
    at vl (chunk-vendors.4f2c8e9b.js:2:92341)
    at yl (chunk-vendors.4f2c8e9b.js:2:91234)
    at bl (chunk-vendors.4f2c8e9b.js:2:91456)
    at Nl (chunk-vendors.4f2c8e9b.js:2:89012)
    at t.unstable_runWithPriority (chunk-vendors.4f2c8e9b.js:2:109876)
    at ka (chunk-vendors.4f2c8e9b.js:2:54321)
    at xl (chunk-vendors.4f2c8e9b.js:2:87654)
    at ul (chunk-vendors.4f2c8e9b.js:2:87543)

Console also shows:
Warning: Cannot update a component while rendering a different component. To locate the bad setState() call inside `UserDashboard`, follow the stack trace as described in https://reactjs.org/link/setstate-in-render
    in UserProfile (at Dashboard.jsx:245)
    in div (at Dashboard.jsx:230)
    in UserDashboard (at App.jsx:89)
    in Route (at App.jsx:87)
    in Switch (at App.jsx:45)
    in Router (at App.jsx:40)

How do I debug minified production errors? Should I enable source maps?""",
            "SKIP_TECHNICAL",
            "complex_react_error_multiline"
        ),
        
        # COMPLEX CONVERSATIONAL TESTS (Should NOT Skip)
        
        # Personal Story with Work Context
        ("""I had such a weird experience at work today that I need to process. I'm a software engineer at a fintech startup in San Francisco, and we just went through our Series B funding round.

So our CTO announced in the all-hands meeting that we're pivoting our entire architecture from a monolithic Rails app to microservices. That's fine, I actually think it's a good move. But here's the thing - he wants to do it in 3 months, right before our planned product launch.

I raised my hand and asked about the risks, you know, like data migration, increased complexity, the learning curve for the team. I tried to be diplomatic about it. But he basically shut me down in front of everyone, saying I was "being negative" and "not a team player."

The thing is, half the team agreed with me in private afterward. My manager even pulled me aside and said "you're right, but you need to pick your battles." But like... isn't this exactly the battle to pick? If we rush this and something breaks in production, that's on us as the engineering team.

I'm only 2 years out of college and this is my first startup job. Maybe I'm wrong? Maybe I'm too idealistic about doing things the right way? I don't want to be known as the person who shoots down ideas, but I also don't want to just stay quiet when I see potential problems.

My girlfriend thinks I should start looking for a new job, but I actually like the product we're building and most of my coworkers. It's just this one situation that's bothering me.

What would you do in my position? Am I overreacting?""",
            None,
            "complex_conversational_work_dilemma"
        ),
        
        # Family Situation with Multiple Concerns
        ("""I need some perspective on a complicated family situation. My parents are both in their early 70s and still living independently in Ohio, while I'm in Seattle with my family.

Here's what's happening: My dad has early-stage Alzheimer's. It's not severe yet - he can still drive, manage most daily tasks, recognize everyone. But it's progressing. My mom has been his primary caregiver, but she's also dealing with her own health issues (diabetes and arthritis).

Last week, my mom called me crying because dad got lost driving home from the grocery store. He ended up 30 miles away in a town he didn't recognize. A police officer helped him get home, but it really scared both of them.

My younger brother lives 20 minutes from them, but he's honestly not very involved. He visits maybe once a month, doesn't really help with doctor's appointments or anything. When I bring it up, he says he's "too busy with work" (he's a real estate agent).

I've been researching options:
- In-home care aids (expensive, and dad might resist having a stranger in the house)
- Assisted living facilities (would mean selling their house, huge emotional impact)
- One of them could move in with me (but I have two kids in school, and my house isn't really set up for elderly care)
- Maybe I need to move back to Ohio? (But my wife has a great job here, kids are settled, I'd be giving up my career)

I feel guilty being so far away. I visit every 2 months, we video call twice a week, but it's not enough. At the same time, I have my own family to think about. My wife is supportive but I can tell she's worried about the financial and emotional burden this might put on us.

My 14-year-old daughter is particularly close to her grandparents. She's upset about grandpa's condition and has been researching Alzheimer's on her own, which I think is stressing her out.

How do people handle this? Is there a right answer? I feel like whatever I choose, I'm letting someone down.""",
            None,
            "complex_conversational_family_caregiving"
        ),
        
        # Career Transition with Personal History
        ("""I'm at a major crossroads in my career and need perspective. I'm 38, married with two kids (8 and 11), working in corporate marketing for 15 years. I'm now a Senior Marketing Director making $145K at a pharma company. The job is stable with excellent benefits, but I'm completely burned out and unfulfilled.

During lockdown, I started learning Python and data science. Over 2 years, I've built projects: web scraper for real estate data, ML model for customer churn prediction, and sentiment analysis tools. I'm genuinely passionate - I code early mornings and spend weekends on Kaggle competitions. For the first time in years, I'm excited about learning.

I'm considering a Master's in CS or Data Science, but it's complicated:

Financial: Programs cost $40-80K, I'd lose income or take 3-4 years part-time, we have mortgage and kids' college savings, my wife (teacher, $55K) would need to support us.

Family: My wife is supportive but worried about money, kids need parental involvement, my mom says I'm having a "midlife crisis."

Career: I'd be 40+ starting over, age discrimination is real, tech layoffs are brutal, maybe I'm romanticizing coding as a hobby vs job.

But I think: If not now, when? In 10 years I'll regret not trying. My therapist says figure out what I value most, but I value financial security AND fulfillment AND family time - they feel mutually exclusive.

What would you do? Has anyone made a career pivot like this in their late 30s?""",
            None,
            "complex_conversational_career_crisis"
        ),
        
        # Personal Health Journey
        ("""I've been struggling with my mental health lately and I really need to talk about it. I work in tech as a senior developer, and the pressure has been building for months. My company just went through another round of layoffs, and even though I kept my job, the uncertainty has me constantly anxious. 

I wake up every morning with this knot in my stomach, worried about whether I'll be next. The work itself is demanding - we're understaffed and the deadlines are unrealistic. I find myself working 12-hour days just to keep up, and when I get home, I'm too exhausted to do anything but scroll through social media and eat takeout.

The worst part is that I'm starting to resent my hobbies. I used to love coding personal projects on weekends, but now it just reminds me of work. My relationships are suffering too. My partner says I never want to go out anymore, and my friends have stopped inviting me to things because I always say no.

I know I need to make changes, but I'm scared. What if I quit and can't find another job? What if I talk to my manager and they think I'm not committed? I've been seeing a therapist for a few weeks, which helps a bit, but I still feel like I'm drowning.

Have you ever been through something similar? How did you cope? I'm open to any advice, even if it's just small things I can do to feel more in control.""",
            None,
            "complex_conversational_mental_health"
        ),
        
        # Family Dynamics and Aging Parents
        ("""My parents are getting older and I'm worried about how to handle their care. My dad is 78 and has early dementia - he gets confused about dates and forgets where he put things, but he still drives and manages most daily tasks. My mom is 76 and has arthritis so bad that she struggles with basic chores like cooking and laundry. They live in a two-story house that's becoming harder for them to manage.

I've been helping more - I go over there every weekend to mow the lawn, do grocery shopping, and help with bills. But I live 45 minutes away and work full-time with two kids of my own. My wife has been amazing about it, but I can tell she's getting frustrated that our weekends are consumed by this.

The real issue is that my parents are in denial. When I suggest they might need help, my dad gets angry and says he's "not ready for assisted living" and my mom changes the subject. They raised me to be independent, and now they can't accept that they need assistance.

I found a local agency that does in-home care, but it's expensive - $3000/month. Their savings are okay but not unlimited. I offered to help pay for it, but they refused, saying they don't want to be a burden.

My brother lives across the country and only calls once a month. When I bring it up, he says "you're closer, you handle it." But I don't want to make all the decisions alone.

How do people navigate this? At what point do you step in even when they resist? I love them so much and just want them to be safe and comfortable, but I feel like I'm failing at balancing everything.""",
            None,
            "complex_conversational_aging_parents"
        ),
        
        # Career Transition with Family Impact
        ("""I'm considering a major career change but it's complicated because of my family situation. I'm 42, married with three kids (ages 8, 11, and 14), and I've been in sales for 18 years. I make good money - $120K base plus commissions, great benefits, stable job. But I'm miserable. The constant pressure to hit quotas, the rejection, the travel... it drains me.

What I really want to do is teach. I have a master's in education from before I switched careers, and I loved working with students. There's an opening at a local community college for an adjunct professor in business communications. It would be part-time, maybe 12 hours a week, and pay about $3K per course.

My wife is a teacher too, so she understands the passion, but she's worried about the financial impact. We'd lose about $40K in income, and our savings aren't huge. We have college funds for the kids, mortgage, and the usual expenses.

The kids are at ages where they need stability. My oldest is starting high school and dealing with all the typical teenager stuff. How do I explain to them that daddy's changing jobs and we might have to cut back on some things?

I could keep my sales job and teach part-time on the side, but that would mean even less family time. I'm already missing too many school events and soccer games.

My therapist says I need to think about what I want my kids to learn from me. Do I want them to successful but unhappy, or fulfilled but making sacrifices?

I have enough experience that I could probably find another sales job if this doesn't work out, but I don't want to look back in 10 years and regret not trying.

What would you do? How do you know when it's the right time to make a big change?""",
            None,
            "complex_conversational_career_family"
        ),
        
        # Friendship and Social Isolation
        ("""I think I'm losing my best friend and I don't know what to do about it. We've been friends for 15 years, since college. We used to talk every day, hang out weekly, travel together. But over the past year, things have changed.

It started when she got a new job that required a lot of travel. She was gone 2-3 weeks per month, and when she was home, she was exhausted and just wanted to rest. I understood at first - I know how demanding work can be.

But now she's back to working remotely from home, and we're still not connecting. When I call, she says she's busy with work or family stuff. When I suggest getting together, she says "maybe next week" but then cancels.

I miss her. I miss having someone who knows all my history, who I can be completely myself with. I've tried new hobbies and joined clubs to meet people, but it's not the same as having that one deep friendship.

I wonder if I'm being too needy. Maybe she's going through something she doesn't want to talk about. Or maybe our lives have just grown apart naturally. People change, circumstances change.

But it hurts. I feel like I'm grieving a loss, but she's still alive and living her life. I see her Instagram posts and she looks happy with other friends.

Should I have a direct conversation about it? Say something like "I feel like we're drifting apart and I miss our friendship"? Or would that make things awkward and push her further away?

I'm scared of losing her completely, but I'm also tired of being the one who always reaches out. How do you know when to let go of a friendship?""",
            None,
            "complex_conversational_friendship_loss"
        ),
        
        # Parenting Teenagers
        ("""My 16-year-old daughter is going through a really tough time and I don't know how to help her. She was always such a happy, outgoing kid - straight A's, lots of friends, involved in sports and clubs. But this year everything changed.

It started with social media. She got into TikTok and Instagram heavily, and now she's obsessed with followers and likes. She spends hours in her room scrolling, and when I try to limit screen time, we have huge fights.

Her grades have slipped from A's to B's and C's. She missed a big soccer tournament because she "forgot" about it while online. Her coach called me concerned.

The worst part is her self-esteem. She compares herself to influencers and models online, and now she hates her body, her face, everything. She asked about getting fillers for her lips last week - she's SIXTEEN. I told her absolutely not, but she was furious.

She won't talk to me about any of this. If I try to bring it up, she says "you don't understand" and storms off. My husband thinks I'm overreacting, that it's just a phase. But I remember being a teenager and how much I struggled with self-image.

I've tried everything: family therapy (she refused to go), taking away her phone (led to a week of silent treatment), having open conversations (she shuts down).

Her friends seem to be going through similar things, but their parents aren't as involved. I don't want to be the strict parent who ruins her social life, but I also can't stand by and watch her hurt herself.

How do other parents handle this? When do you push for professional help? I'm worried I'm making it worse by being too controlling.""",
            None,
            "complex_conversational_teenage_daughter"
        ),
        
        # Financial Stress and Adulting
        ("""I'm 28 and I feel like I'm constantly one emergency away from financial disaster. I make $75K as a software engineer, which should be plenty, but somehow I'm always stressed about money.

My apartment rent is $2200/month in San Francisco - that's 35% of my take-home pay. Then there's my car payment ($450), student loans ($380), credit card minimums ($200), and all the other stuff. I have about $5000 in savings, which feels like nothing.

The problem is lifestyle creep. When I got my raise last year, I upgraded my phone, started eating out more, bought a nicer car. Now I can't cut back because I "need" these things.

I have no idea how to budget. I try apps like Mint and YNAB, but I get overwhelmed and stop using them. I know I should cook more and cancel subscriptions, but I'm exhausted after work and just want to order DoorDash.

My friends are all in similar situations - good jobs, high cost of living, no financial buffer. We talk about it sometimes, but no one has solutions.

I'm single, so no partner to share expenses with. My parents help occasionally, but they're retired and I don't want to burden them.

I want to buy a house someday, go on vacations, have kids, retire early. But right now, I feel trapped. Every time I think about my finances, I get anxious and want to scroll social media to distract myself.

How do people get out of this cycle? Should I see a financial advisor? Move to a cheaper city? I feel like I'm too old to be this bad with money.""",
            None,
            "complex_conversational_financial_stress"
        ),
        
        # Long-distance Relationship Challenges
        ("""My boyfriend and I have been in a long-distance relationship for 2 years, and lately I'm starting to wonder if it's worth it. We met in college, dated for a year, then he moved across the country for grad school. I stayed local for my job.

We see each other every 6-8 weeks, which means a lot of weekends are spent traveling or recovering from travel. Flights are expensive ($400 round trip), hotels add up, and the time difference means our schedules never align perfectly.

The distance is changing us. We used to talk for hours every day, but now conversations feel forced. We run out of things to say, or we argue about stupid things because we're tired and stressed.

I'm lonely here. My friends have coupled up and have less time for me. I go to weddings and events alone, which makes me feel like the perpetual third wheel.

He says he loves me and wants to be together eventually, but "eventually" could be years. His program ends in 18 months, but then what? Would he move here? Would I move there? We both have established lives and careers.

I think about breaking up sometimes. It would hurt, but maybe I'd be happier. Then I remember all the good times, how much I love him, how he makes me laugh like no one else.

My therapist says long-distance relationships work for some people and not others, and I need to decide what I can sustain. But how do you know? How do you weigh the pain of distance against the fear of regret?

I don't want to throw away something good because it's hard. But I also don't want to waste years of my life waiting for "someday."

What would you do? Have you been in a similar situation?""",
            None,
            "complex_conversational_long_distance"
        ),
        
        # Creative Block and Identity Crisis
        ("""I'm a 35-year-old graphic designer and I think I'm having a midlife crisis. I've been in this field for 12 years, and I used to love it. I was passionate about branding, typography, visual storytelling. My work won awards, clients loved me.

But now? I dread opening my laptop. Every project feels the same - another logo, another website, another social media graphic. The industry has changed so much with AI tools and stock templates. Everything feels commoditized.

I look at my portfolio and see technically good work, but it doesn't excite me anymore. I don't feel like I'm creating anything meaningful. I scroll through Dribbble and Behance and feel inadequate - everyone else seems so innovative and on-trend.

I think about switching careers completely. Maybe teaching design, or UX research, or even something totally different like writing or photography. But I'm scared. I have a mortgage, a family to support. Starting over at 35 feels terrifying.

My wife says I should take a sabbatical, travel, find inspiration. But that feels selfish and expensive. My friends say it's normal, that everyone hits this wall eventually.

I miss that feeling of flow, when time disappears and you're completely absorbed in your work. I miss feeling proud of what I create.

Is this just burnout? Should I push through it? Or is it time for a change? How do you know when your passion has run its course?

I don't want to look back at 50 and regret not trying something new. But I also don't want to impulsively quit and ruin my family's stability.""",
            None,
            "complex_conversational_creative_block"
        ),
    ]
    
    # Conversational Tests (Should NOT be skipped)
    CONVERSATIONAL_TESTS = [
        (
            "My wife Sarah loves hiking and mystery novels, what should I get her for our anniversary?",
            None,
            "conversational_family"
        ),
        (
            "I work as a senior software engineer at Tesla and I'm struggling with imposter syndrome",
            None,
            "conversational_career"
        ),
        (
            "I'm learning Spanish because I'm moving to Barcelona next year",
            None,
            "conversational_learning"
        ),
        (
            "My son is having trouble with math at school, how can I help him?",
            None,
            "conversational_family_help"
        ),
        (
            "I have a golden retriever named Max and he's been acting anxious lately",
            None,
            "conversational_pet"
        ),
        (
            "I grew up in a small town in Ohio and moved to New York for my career",
            None,
            "conversational_background"
        ),
        (
            "I'm allergic to peanuts and need to be careful when eating out",
            None,
            "conversational_health"
        ),
        (
            "I love playing guitar in my free time, been doing it for 10 years",
            None,
            "conversational_hobby"
        ),
        (
            "I'm vegetarian and looking for healthy dinner recipes with lots of vegetables",
            None,
            "conversational_preferences"
        ),
        (
            "I'm considering switching careers from marketing to software development",
            None,
            "conversational_goals"
        ),
        (
            "I'm having trouble understanding React hooks in my new job at the startup",
            None,
            "conversational_work_challenge"
        ),
        (
            "My daughter is interested in learning piano, what's a good starting age?",
            None,
            "conversational_parenting"
        ),
        
        # REALISTIC CONVERSATIONAL INTERACTIONS (semantic path - should NOT skip)
        
        # Questions about personal preferences
        (
            "I'm thinking about getting into photography as a hobby. I've always loved taking pictures on my phone but want to step up my game. Should I start with a DSLR or mirrorless camera? I'm willing to spend around $1000 to $1500.",
            None,
            "realistic_hobby_advice"
        ),
        
        # Seeking recommendations with personal context
        (
            "I'm planning a surprise birthday party for my wife's 40th next month. She loves Italian food and wine, and we have about 20 close friends invited. Do you have any suggestions for fun party games or activities that aren't too cheesy?",
            None,
            "realistic_event_planning"
        ),
        
        # Career advice with personal situation
        (
            "I got two job offers and I'm torn between them. One is a senior role at a stable Fortune 500 company with great benefits, the other is a lead position at a startup with equity but more risk. I have a family to support but I'm also excited about the growth potential at the startup. How do I make this decision?",
            None,
            "realistic_career_decision"
        ),
        
        # Personal health journey
        (
            "I've been trying to lose weight for months now. I'm 35 pounds overweight and it's affecting my energy levels and confidence. I've tried keto, intermittent fasting, and counting calories, but I always fall back into old habits after a few weeks. What's worked for you or people you know?",
            None,
            "realistic_health_journey"
        ),
        
        # Relationship advice
        (
            "My best friend is going through a divorce and she's been staying with me for the past two weeks. I want to be supportive, but it's starting to strain my own marriage. My husband is patient but I can tell he's getting frustrated. How do I balance being a good friend without neglecting my relationship?",
            None,
            "realistic_relationship_balance"
        ),
        
        # Parenting challenges
        (
            "My 13-year-old son has been increasingly withdrawn lately. He used to be social and talkative, but now he just stays in his room playing video games. His grades are still okay, but I'm worried this might be early signs of depression. Should I be concerned or is this just normal teenage behavior?",
            None,
            "realistic_parenting_concern"
        ),
        
        # Home improvement with context
        (
            "We just bought our first house and the backyard is a mess - overgrown grass, weeds everywhere, and the soil seems really poor. I have no gardening experience but I'd love to create a nice space for my kids to play. Where do I even start? Should I hire a landscaper or try to learn it myself?",
            None,
            "realistic_home_project"
        ),
        
        # Financial planning question
        (
            "I'm 32 and just realized I haven't been saving for retirement properly. I have about $15k in a 401k from my previous job and nothing else. I make $75k a year. How worried should I be, and what steps should I take now to get on track?",
            None,
            "realistic_financial_advice"
        ),
        
        # Travel planning with constraints
        (
            "I'm planning a family vacation to Europe next summer - me, my wife, and our two kids (ages 10 and 13). We have two weeks and want to see London, Paris, and maybe one more city. Is that too ambitious? Any tips for traveling with kids internationally for the first time?",
            None,
            "realistic_travel_planning"
        ),
        
        # Learning motivation
        (
            "I've always wanted to learn to code, and I finally have some free time. I'm 45 and work in marketing, but I find the logic and problem-solving aspect fascinating. Am I too old to start? And where should a complete beginner begin - Python, JavaScript, something else?",
            None,
            "realistic_learning_motivation"
        ),
        
        # Pet adoption decision
        (
            "My kids have been begging for a dog for years, and we're finally considering it. We live in a house with a fenced yard, both parents work from home 3 days a week. We've never had a pet before. Should we get a puppy or adopt an older dog? And what breed would be good for first-time owners with young kids?",
            None,
            "realistic_pet_decision"
        ),
        
        # Workplace navigation
        (
            "I've been at my company for 3 years and I think I deserve a promotion, but my manager keeps saying 'not yet' without giving specific feedback. My colleague who started after me just got promoted. I'm starting to feel undervalued. Should I push harder for the promotion or start looking elsewhere?",
            None,
            "realistic_workplace_navigation"
        ),
        
        # Hobby recommendation
        (
            "I'm feeling really burned out from work and need a hobby that helps me relax and disconnect. I'm not very artistic or crafty, but I like being outdoors and I enjoy problem-solving. Any suggestions for hobbies that might fit?",
            None,
            "realistic_hobby_search"
        ),
        
        # Life transition
        (
            "I'm about to become an empty nester - my youngest is going to college in the fall. After 20 years of being 'mom' as my primary identity, I'm not sure who I am anymore. How do people cope with this transition? I'm excited for her but also feeling a bit lost.",
            None,
            "realistic_life_transition"
        ),
        
        # Social situation
        (
            "I have a coworker who keeps inviting me to social events outside work, but I'm more of an introvert and I prefer to keep work and personal life separate. I don't want to be rude, but I also don't want to encourage more invitations. How do I handle this politely?",
            None,
            "realistic_social_boundary"
        ),
        
        # Mixed technical with personal learning
        (
            "I'm trying to build a simple website for my small business (I'm a personal trainer). I know nothing about web development. Should I use Wix or Squarespace, or is it worth learning HTML/CSS? I want something professional-looking but I don't have a huge budget.",
            None,
            "realistic_business_tech"
        ),
        
        # Asking for explanation of concepts
        (
            "Can you explain how blockchain actually works? I keep hearing about it in the news and my nephew won't stop talking about crypto, but I don't really understand what makes it different from regular databases or why everyone thinks it's revolutionary.",
            None,
            "realistic_concept_explanation"
        ),
        
        # Decision-making with trade-offs
        (
            "My elderly parents want to stay in their home, but it's becoming difficult for them to manage. The house has stairs, the yard needs maintenance, and they're 45 minutes from me. I could move them closer or into assisted living, but they're very resistant to change. What's the best approach?",
            None,
            "realistic_family_decision"
        ),
        
        # Seeking validation
        (
            "Am I wrong for being upset that my sister-in-law constantly gives parenting advice? She doesn't have kids of her own, but she always has opinions about how I'm raising mine. I've tried to be polite, but it's really getting under my skin. Is it okay to tell her to back off?",
            None,
            "realistic_family_conflict"
        ),
        
        # Future planning
        (
            "I'm thinking about going back to school to get my MBA, but I'm 38 with a family and a full-time job. The program would be part-time over 3 years. Is it worth it at my age? Will employers even care about an MBA from someone who got it later in their career?",
            None,
            "realistic_education_planning"
        ),
        
        # MORE REALISTIC CONVERSATIONAL INTERACTIONS (26 new diverse scenarios)
        
        # Neurodiversity and parenting
        (
            "My 9-year-old son was just diagnosed with ADHD and I'm feeling overwhelmed. The doctor prescribed medication but I'm worried about side effects. He's such a bright, creative kid and I don't want to change who he is. But he's struggling in school and his teachers keep calling me about his behavior. How do I support him without trying to 'fix' him? Other parents with ADHD kids, what helped?",
            None,
            "realistic_adhd_parenting"
        ),
        
        # Cultural identity and immigration
        (
            "I moved to the US from South Korea when I was 12, and now I'm 28 with a 3-year-old daughter. I want her to learn Korean and understand her heritage, but my husband (who's American) doesn't speak the language. We live in a suburb with very few Asian families. I feel like I'm caught between two worlds and don't fully belong to either. How do I pass on my culture to her when I'm still figuring out my own identity?",
            None,
            "realistic_cultural_identity"
        ),
        
        # Chronic illness management
        (
            "I was diagnosed with multiple sclerosis two years ago and I'm learning to accept my new normal. Some days I'm fine, other days I can barely get out of bed. My employer has been understanding but I can tell they're frustrated when I have to call in sick. My friends don't really get it - they see me on good days and think I'm exaggerating. I'm only 34 and feel like my life is already over. How do people with chronic illnesses find meaning and purpose?",
            None,
            "realistic_chronic_illness"
        ),
        
        # Creative career struggles
        (
            "I'm a graphic designer freelancing for 3 years now. I love the creative freedom but the income instability is killing me. Some months I make $8k, others barely $2k. I can't get a mortgage or plan for the future. My parents keep asking when I'll get a 'real job' and honestly, I'm starting to wonder the same thing. But the thought of going back to a 9-to-5 at an agency makes me feel trapped. Is there a middle ground?",
            None,
            "realistic_freelance_struggle"
        ),
        
        # LGBTQ+ family planning
        (
            "My wife and I (we're both women) are trying to start a family through IVF. We're on our third round and it's emotionally and financially draining. Each failed cycle feels like a loss. We've spent $45k already and our savings are almost gone. Some family members don't support us having kids and it hurts. I'm starting to question if we should keep trying or consider adoption. How do you know when to stop trying?",
            None,
            "realistic_lgbtq_family_planning"
        ),
        
        # Entrepreneurship fear
        (
            "I've been developing a SaaS product in my spare time for 18 months. I have 50 beta users who love it and 10 are paying. My day job pays $95k and I have good benefits. Part of me wants to quit and go all-in on my startup, but I have a mortgage and my wife is pregnant with our second child. The safe choice is stay employed. The dream is to build something of my own. I'm 33 - is this my last chance to take a risk?",
            None,
            "realistic_entrepreneur_fear"
        ),
        
        # Grief and moving forward
        (
            "My mom died from cancer 8 months ago and I'm not handling it well. Everyone says 'time heals' but I feel worse now than at the funeral. I dream about her constantly. Small things trigger me - her favorite song, the perfume she wore. I have a therapist but talking doesn't bring her back. My dad is dating someone new already and I'm angry at him, which isn't fair. How long does this hurt last?",
            None,
            "realistic_grief_process"
        ),
        
        # Minimalism and lifestyle change
        (
            "I'm considering selling almost everything I own and traveling for a year. I'm 29, single, no kids, and I've been working in consulting since college. I have $65k saved and no debt. Everyone thinks I'm crazy - 'what about your career?' But I feel like I'm sleep-walking through life. I want to see the world before I settle down. Is a career gap really that bad? Will I regret this in 10 years?",
            None,
            "realistic_minimalism_travel"
        ),
        
        # Social media and mental health
        (
            "I deleted Instagram 3 weeks ago and it's been harder than I expected. I keep reaching for my phone to check it out of habit. I feel less anxious but also more isolated - like I'm missing out on what my friends are doing. I started because I was comparing myself to everyone constantly and it was affecting my self-esteem. But now I wonder if I'm being too extreme. Can you use social media in a healthy way or is quitting the only option?",
            None,
            "realistic_social_media_detox"
        ),
        
        # Sandwich generation caregiving
        (
            "I'm 42 with three kids (ages 8, 10, 14) and both my parents need increasing help. My mom has dementia and my dad just had a heart attack. My brother lives across the country and is basically useless. I'm working full-time, managing my kids' schedules, and now I'm essentially parenting my parents too. I haven't had a full night's sleep in months. My wife is supportive but she's overwhelmed too. I don't know how much longer I can do this without breaking.",
            None,
            "realistic_sandwich_generation"
        ),
        
        # Body image and fitness
        (
            "I've lost 60 pounds over the past year through diet and exercise. I should be happy but I hate how I look - the loose skin, the stretch marks. I worked so hard and I still don't feel attractive. My husband says I look great but I think he's just being nice. I'm considering plastic surgery but it's expensive and I feel vain for even thinking about it. When does the mental transformation catch up with the physical one?",
            None,
            "realistic_body_image"
        ),
        
        # Academic pressure and perfectionism
        (
            "I'm a junior in high school and I'm so tired of the pressure. I have a 4.0 GPA, I'm in 4 AP classes, I play varsity soccer, and I volunteer at the hospital. My parents expect me to get into an Ivy League school. But I'm burned out. I had a panic attack during my SATs. I don't even know what I want to study - I'm just doing what everyone expects. Is it okay to just be average? To go to a state school and have a normal life?",
            None,
            "realistic_academic_pressure"
        ),
        
        # Recovering from addiction
        (
            "I've been sober from alcohol for 6 months and it's the hardest thing I've ever done. I lost my marriage, my job, and almost my relationship with my kids because of drinking. I'm rebuilding slowly - new job, small apartment, supervised visits with my kids. But I'm so lonely. All my old friends were drinking buddies. I go to AA but I don't really connect with anyone. How do you make friends as an adult when you can't go to bars or parties?",
            None,
            "realistic_addiction_recovery"
        ),
        
        # Non-traditional education path
        (
            "I dropped out of college after one semester and went to a coding bootcamp instead. That was 5 years ago and I'm now making $110k as a software engineer. But I feel insecure around my friends who have degrees. My company wants to promote me to lead but some of my team members have computer science PhDs. Am I an impostor? Should I go back and finish my degree even though I don't need it for my career?",
            None,
            "realistic_nontraditional_education"
        ),
        
        # Interracial relationship challenges
        (
            "I'm Black and my fiancée is white. We're planning our wedding and both families are being difficult in different ways. Her parents keep making 'jokes' that are actually microaggressions. My mom is upset I'm not marrying a Black woman. We love each other but the family drama is exhausting. We're considering just eloping but that feels like letting the negativity win. How do interracial couples navigate these issues?",
            None,
            "realistic_interracial_relationship"
        ),
        
        # Climate anxiety and lifestyle
        (
            "I'm becoming paralyzed by climate anxiety. Every decision feels loaded - should I have kids when the planet is dying? Is my job in oil and gas morally wrong? Should I sell my car? I recycle and buy sustainable products but it feels like drops in the ocean. My friends think I'm overreacting but the science is terrifying. How do you take climate change seriously without falling into despair?",
            None,
            "realistic_climate_anxiety"
        ),
        
        # Blended family dynamics
        (
            "I married a man with two kids from his previous marriage (ages 7 and 9). I'm trying to be a good stepmom but it's so hard. They visit every other weekend and I feel like a guest in my own home. Their mom bad-mouths me to them. My husband expects me to love them like they're mine but I barely know them. I want kids of my own but I'm worried about how that will affect the family dynamic. Is it normal to not instantly love your stepkids?",
            None,
            "realistic_blended_family"
        ),
        
        # Financial independence and early retirement
        (
            "I'm 28 and I've been following FIRE (Financial Independence Retire Early) for 3 years. I save 65% of my income, live with roommates, bike to work, and rarely go out. I'll be able to retire at 40. But I'm missing out on my 20s. My friends travel, go to concerts, date. I feel like I'm sacrificing the best years of my life for a future that's not guaranteed. Am I being too extreme? Is there a balanced approach to FIRE?",
            None,
            "realistic_fire_movement"
        ),
        
        # Religious deconstruction
        (
            "I was raised in a very conservative Christian household. I went to church 3 times a week, Christian schools, the works. Now at 26, I'm questioning everything. I don't think I believe anymore but I haven't told my family. If they find out, I'll lose my whole community. My parents might disown me. I feel like I'm living a double life. How do you leave a religion when it's your entire identity and support system?",
            None,
            "realistic_religious_deconstruction"
        ),
        
        # Career vs passion conflict
        (
            "I'm a corporate lawyer making $180k. I hate it. I went to law school because my immigrant parents sacrificed everything for my education. But I've always wanted to be a music teacher. I play piano and I volunteer teaching kids on weekends - it's the only time I'm truly happy. But switching careers means a huge pay cut and disappointing my parents. I'm 31 - am I too old to completely change paths? Do I owe it to my parents to keep doing this?",
            None,
            "realistic_career_passion_conflict"
        ),
        
        # Neurodivergent adult diagnosis
        (
            "I was just diagnosed with autism at 35 and everything suddenly makes sense. Why I've always felt different, why social situations exhaust me, why I hyperfocus on specific interests. Part of me is relieved to have an explanation. But I'm also grieving - grieving the person I thought I was, grieving all the years I spent trying to be 'normal.' Do I tell people? Will they treat me differently? How do I unmask after decades of forcing myself to fit in?",
            None,
            "realistic_adult_autism_diagnosis"
        ),
        
        # Childless by choice judgment
        (
            "My husband and I decided not to have kids. We're both 34 and happy with our choice. But everyone from family to coworkers to strangers feels entitled to tell us we'll change our minds or that we're selfish. My mom cries about never having grandkids. People say our lives lack purpose. I'm tired of defending a deeply personal decision. Why is it so hard for people to accept that some people genuinely don't want children?",
            None,
            "realistic_childfree_by_choice"
        ),
        
        # Digital nomad isolation
        (
            "I've been working remotely and traveling for 2 years. Bali, Thailand, Portugal, Mexico. Instagram makes it look glamorous but I'm really lonely. I have work friends on Zoom but no real friends. Every time I start to connect with someone, I move to the next place. I'm 31 and wondering if I've wasted my late 20s chasing an idealized lifestyle. When do you stop running and actually build a life somewhere?",
            None,
            "realistic_digital_nomad_lonely"
        ),
        
        # Recovering from financial abuse
        (
            "I left an abusive relationship 8 months ago. He controlled all the money - I had no bank account, no credit cards, no credit history of my own. I'm 36 and starting from zero. I'm living with my sister, working retail, trying to rebuild my credit. I'm embarrassed to be starting over at this age. I see women my age buying houses and saving for retirement. Will I ever catch up financially? How do you rebuild your life after this kind of abuse?",
            None,
            "realistic_financial_abuse_recovery"
        ),
        
        # Imposter syndrome in non-traditional field
        (
            "I'm a self-taught photographer running a successful wedding photography business. I charge $3-5k per wedding and I'm booked solid. But I constantly feel like a fraud. I didn't go to art school. I don't know all the technical terms. Other photographers talk about lenses and lighting setups and I just... figure it out on the job. Clients love my work but I'm terrified someone will call me out as a fake. When does the imposter syndrome go away?",
            None,
            "realistic_imposter_creative_field"
        ),
        
        # Ethical consumption struggles
        (
            "I'm trying to be more ethical in my consumption but it's overwhelming and expensive. I want to buy fair trade, sustainable, cruelty-free everything. But I'm a single mom making $45k. I can't afford the $80 organic cotton t-shirt. I shop at thrift stores when I can but sometimes I need something new and fast fashion is all I can afford. I feel guilty but also resentful. Why is ethical consumption only accessible to the wealthy?",
            None,
            "realistic_ethical_consumption"
        ),
        
        # New conversational tests with unique topics, profiles, and contexts
        
        # Career transition for a parent
        (
            "I'm a 38-year-old father of two who's been working in banking for 12 years, but I want to transition to teaching high school math. My wife is supportive but worried about the significant pay cut. I'd go from $110k to maybe $45k. My kids are 7 and 10, so there are college funds to consider. I miss the summer breaks from teaching like I had when I was subbing in college. Is this a realistic career change?",
            None,
            "realistic_parent_career_change"
        ),
        
        # Mental health support for student
        (
            "I'm a college sophomore struggling with anxiety and depression. I've been in therapy for 6 months but still have days where I can't get out of bed. My roommate doesn't understand and thinks I'm just being lazy. I dropped from 15 credits to 9 because I was overwhelmed. My parents think I'm not trying hard enough. How do I manage school while dealing with this?",
            None,
            "realistic_student_mental_health"
        ),
        
        # Small business owner challenges
        (
            "I started a bakery 3 years ago and it's growing, but I'm completely burned out. I'm working 70-hour weeks, my husband feels like he's married to a ghost, and I've lost 20 pounds from stress. I have 5 employees now, and they depend on me for their income. I want to scale back, but I feel trapped. Should I sell the business or hire a manager?",
            None,
            "realistic_small_business_owner"
        ),
        
        # Elderly care with family conflict
        (
            "My 82-year-old father has been diagnosed with Alzheimer's and needs full-time care, but my brother and I can't agree on how to handle it. He thinks I should quit my job to care for Dad, while I think we should look into assisted living. My brother lives 2 hours away and only visits once a month, but he wants to make all the decisions. Our father would hate being in a facility, but I can't manage this alone.",
            None,
            "realistic_elderly_care_conflict"
        ),
        
        # First-time homebuyer confusion
        (
            "My partner and I are first-time homebuyers with a $500k budget in Seattle. We've been looking for 6 months and keep getting outbid. Our realtor suggests offering 10-15% over asking, but that would blow our budget. We found a place we love, but it needs $30k in repairs. Should we offer over asking, or wait longer? We feel like we're missing obvious signs.",
            None,
            "realistic_homebuyer_confusion"
        ),
        
        # Tech worker questioning company values
        (
            "I'm a senior engineer at a social media company that recently made changes that prioritize engagement over user wellbeing. Features designed to increase screen time are affecting my own family - my teenage sister is becoming addicted to the platform. I'm making good money, but I'm struggling with ethical concerns. Should I speak up internally or look for a new job?",
            None,
            "realistic_ethical_tech_worker"
        ),
        
        # Graduate student debt and career prospects
        (
            "I just completed my master's in social work with $80k in debt. Starting salaries are $40-45k, and I'm not sure I can afford payments on that salary, especially living in the city where most jobs are. I chose this career because I want to help people, but now I'm questioning if it was financially responsible. Should I consider a different field?",
            None,
            "realistic_graduate_debt_career"
        ),
        
        # Single parent juggling multiple responsibilities
        (
            "I'm a single mom working full-time as a nurse while taking online classes for my RN-to-BSN. My 8-year-old daughter has ADHD and needs constant supervision. I work 3 12-hour shifts, study at night, and barely sleep. I'm failing one of my classes despite trying. How do other single parents balance everything? I feel like I'm failing both my daughter and myself.",
            None,
            "realistic_single_parent_struggle"
        ),
        
        # Returning to work after parental leave
        (
            "I just returned to work after 8 months of parental leave, and everything feels different. My role was covered by someone else who apparently did some things differently. My manager says I need to 'catch up' but I'm still exhausted from sleepless nights. I was a top performer before, but now I feel like I'm starting over. Is this normal? How do I manage the transition?",
            None,
            "realistic_returning_parent"
        ),
        
        # Midlife friendship crisis
        (
            "I'm 34 and my close-knit friend group from college is all getting married and having kids. I'm still single and child-free by choice, but I feel like I don't fit in anymore. Our conversations are all about pregnancy symptoms, daycare options, and sleep schedules. I want to be happy for them, but I feel left out and like I'm missing part of my identity. Is this normal?",
            None,
            "realistic_friendship_identity_crisis"
        ),
        
        # Starting a family vs career goals
        (
            "My husband and I want to start a family, but I just got accepted to my dream graduate program. It's a 3-year program with heavy workload and low funding. We're not getting any younger, but I want this degree since undergrad. If I wait, will I still be eligible when I'm 30? Is it possible to be a good parent and a committed graduate student?",
            None,
            "realistic_family_career_timing"
        ),
        
        # Dealing with toxic workplace
        (
            "I work in a marketing agency where the culture is 'hustle culture' - working 60+ hours is normal, weekend work is expected, and employees are called 'lazy' for taking PTO. My manager yells at people in meetings. I'm good at my job and well-compensated, but I've started having panic attacks about going to work. Should I stay for the financial security or risk my career stability?",
            None,
            "realistic_toxic_workplace"
        ),
        
        # Coping with a chronic illness diagnosis
        (
            "I was recently diagnosed with rheumatoid arthritis at age 29. I used to be very active - running, rock climbing, hiking. Now I'm dealing with joint pain and fatigue. I had to quit my job as a personal trainer because the physical demands are too difficult. I feel like I've lost my identity and I'm worried about my financial future. How do I adjust my life and career?",
            None,
            "realistic_chronic_illness_adjustment"
        ),
        
        # Cultural identity and family expectations
        (
            "I'm a second-generation immigrant trying to honor my family's sacrifices while pursuing my own dreams. My parents expect me to become a doctor like the 'model minority' path, but I'm passionate about art and design. They say I'm wasting my potential and bringing shame to the family. I have a design degree and a job I love, but I feel torn between filial duty and personal fulfillment.",
            None,
            "realistic_cultural_identity_conflict"
        ),
        
        # Relationship challenges with different life goals
        (
            "My boyfriend and I have been together for 5 years, and we're facing a major life goals misalignment. He wants to travel the world for 2 years, working remotely, while I want to settle down and buy a house. We both make good money but in different cities. Neither of us wants to move permanently to the other's city. I love him but I don't know how to reconcile these different life visions.",
            None,
            "realistic_relationship_goals_mismatch"
        ),
        
        # Dealing with infertility
        (
            "My husband and I have been trying to conceive for 2 years with no success. We just finished our first round of fertility treatments which failed. I'm emotionally and financially drained. My friends are getting pregnant easily and I feel guilty for not being happy for them. My family keeps asking when I'll have kids. How do I cope with infertility while maintaining relationships?",
            None,
            "realistic_infertility_journey"
        ),
        
        # Starting over after divorce
        (
            "I just finalized my divorce after 15 years of marriage. I have 2 teenagers who are struggling with the change. I went back to work after years as a stay-at-home parent, but I feel behind in my field. My 401k is gone, I have custody part-time, and I'm starting over at 42. I feel like I'm behind everyone else. How do I rebuild my life and career?",
            None,
            "realistic_divorce_rebuild"
        ),
        
        # Anxiety about climate change
        (
            "I'm 26 and increasingly anxious about climate change. It's affecting my decisions about having kids, where to live, and even my career. I've cut my carbon footprint significantly but feel hopeless about the scale of the problem. I have panic attacks thinking about the future my potential children will face. How do I maintain hope while acknowledging the seriousness?",
            None,
            "realistic_climate_anxiety"
        ),
        
        # Dealing with a difficult boss
        (
            "I have a micromanaging boss who asks for updates multiple times per day and takes credit for my work. I've been passed over for 2 promotions in favor of less qualified colleagues. I've tried talking to HR but nothing changed. I love my team and my role, but I dread going to work. Should I stay and wait for the boss to leave or find something else?",
            None,
            "realistic_difficult_boss"
        ),
        
        # LGBTQ+ identity and family rejection
        (
            "I'm 22 and just came out as transgender to my family. They've completely shut down communication, and I've been cut off from financial support for college. My parents say I'm confused and that they're 'protecting' me from making a mistake. I have a few supportive friends but I'm struggling with depression and financial stress. How do I navigate this transition without family support?",
            None,
            "realistic_lgbtq_family_rejection"
        ),
        
        # Imposter syndrome in leadership role
        (
            "I was recently promoted to department manager, but I feel like I don't belong. I'm younger than most of my team members, and some of them have more experience than me. I constantly doubt my decisions and ask other managers for validation. Imposter syndrome is affecting my confidence and leadership. How do I develop the authority and confidence I need?",
            None,
            "realistic_leadership_imposter_syndrome"
        ),
        
        # Caring for a sibling with addiction
        (
            "My younger brother has been struggling with opioid addiction for 5 years. He's been in and out of treatment programs, and each time he promises it's the last time. My parents enable his behavior, but I think tough love is needed. He's my sibling, but I'm tired of being manipulated and lied to. How do I help without enabling?",
            None,
            "realistic_sibling_addiction"
        ),
        
        # Career plateau and motivation
        (
            "I've been in the same role for 7 years and feel completely stagnant. I'm not being challenged, promotions seem impossible due to company politics, and I'm doing the bare minimum to get by. I'm still early in my career (age 31) but feel like it's too late to switch fields. I want to feel passionate about my work again. How do I reignite motivation?",
            None,
            "realistic_career_plateau"
        ),
        
        # Balancing personal values with career
        (
            "I work in consulting for large corporations where I help them optimize costs, which often means recommending mass layoffs. I make excellent money, but I struggle with the ethical implications. I've been offered a position at a nonprofit with 40% less pay but work that aligns with my values. I have a mortgage and student loans. How do I balance values with financial security?",
            None,
            "realistic_values_vs_finances"
        ),
        
        # Dealing with grief and loss
        (
            "I lost my best friend in a car accident 6 months ago. We talked daily and did everything together. I'm trying to maintain our mutual friendships, but it's painful to be around the memories. I'm not sure how to grieve as an adult - my family thinks I should be 'over it' by now. The grief comes in waves. How do I process this loss?",
            None,
            "realistic_grief_processing"
        ),
        
        # Work-life balance in demanding job
        (
            "I work in investment banking and regularly work 80-hour weeks during deal seasons. I've missed multiple family events and my relationship is struggling. My partner says I'm married to my job. I'm in my thirties and want to enjoy life, but I also want to make partner. I know this job is temporary, but the present costs are high. How do I balance ambition with relationships?",
            None,
            "realistic_work_life_balance"
        ),
        
        # Dealing with chronic pain
        (
            "I've been living with chronic back pain for 3 years after a car accident. It affects my concentration, energy, and mood. I can't do the physical activities I used to love. Many doctors have dismissed my symptoms, and I've had to change jobs. I feel like different person and I'm grieving the life I used to have. How do I adapt to this new reality?",
            None,
            "realistic_chronic_pain_adaptation"
        ),
        
        # Moving back to hometown
        (
            "I'm 35 and moving back to my small hometown after 10 years in a major city for a job opportunity. I sold my condo and left my established friend circle. I feel like I'm going backward in my life. Everyone here had the same life path - high school, local college, marriage, kids. I have different experiences and perspectives. How do I adjust to a smaller life?",
            None,
            "realistic_small_town_return"
        ),
        
        # Managing multiple sclerosis diagnosis
        (
            "I was diagnosed with MS at 28. I'm trying to keep my life stable while managing fatigue, brain fog, and physical symptoms. My company doesn't understand why I need accommodations. My friends still invite me out, but I have to cancel frequently due to symptoms. I want to maintain my independence and social life, but it's challenging. How do I navigate my changing capabilities?",
            None,
            "realistic_ms_diagnosis_management"
        ),
        
        # Dealing with burnout in helping profession
        (
            "I'm a social worker dealing with child abuse cases and I'm experiencing secondary trauma. I'm having nightmares and difficulty connecting with my own children. The caseload is overwhelming, and I feel like I can't make real change. I'm considering leaving the profession I was passionate about. How do I restore my passion while protecting my mental health?",
            None,
            "realistic_career_burnout_helping_profession"
        ),
        
        # Navigating multigenerational workplace
        (
            "I manage a team with age ranges from 22 to 60. I'm 35 and feel caught between different generational expectations. My younger employees want more feedback and growth opportunities, while my older employees want respect for experience. I'm trying to be fair and effective, but I feel like I'm constantly managing generational tensions. How do I lead effectively?",
            None,
            "realistic_multigenerational_management"
        ),
        
        # Dealing with infertility as a man
        (
            "My wife and I are struggling with infertility, and it's affecting me deeply, but I feel like my emotions are dismissed since reproductive issues are often seen as 'women's problems'. I'm watching my wife go through painful treatments while I can't 'fix' this. My friends don't know how to support male infertility. How do I navigate this?",
            None,
            "realistic_male_infertility"
        ),
        
        # Returning to school as adult learner
        (
            "I'm 38 and going back to school to become a nurse. I'm in classes with 18-22 year olds, and I feel completely out of place. I have family obligations that make studying difficult. I work part-time to support my family while in school. I question if I'm too old to start a new career. How do I balance school, family, and age-related challenges?",
            None,
            "realistic_adult_learner_nursing"
        ),
        
        # Coping with empty nest syndrome
        (
            "My youngest just left for college and I'm experiencing unexpected sadness. I've been a full-time mom for 18 years and I don't know who I am outside of parenting. My friends who are still in the thick of parenting don't understand. I have hobbies but they don't fulfill the purpose of my life. How do I rediscover my identity?",
            None,
            "realistic_empty_nest_identity"
        ),
        
        # Dealing with age discrimination in job search
        (
            "I'm 52 and looking for work in marketing, but I feel like my age is working against me. Employers seem interested until they see my age on my resume. They want 'digital natives' and 'fresh perspectives.' I have decades of experience and knowledge of the industry that's changing rapidly. How do I compete with younger candidates?",
            None,
            "realistic_age_discrimination_job_search"
        ),
        
        # Managing anxiety in the workplace
        (
            "I have anxiety that manifests in workplace settings, particularly during presentations and meetings. I've been avoiding leadership opportunities because of this. My colleagues don't know about my anxiety, and I'm worried about stigma. I want to advance my career but fear my anxiety will hold me back. How do I manage this professionally?",
            None,
            "realistic_workplace_anxiety_management"
        ),
        
        # Dealing with perfectionism at work
        (
            "I have perfectionist tendencies that are affecting my productivity and teamwork. I spend too much time on details that don't matter to others. My manager has told me to 'let go of perfect' but I can't seem to do it. I'm missing deadlines because I want everything to be flawless. How do I balance quality with efficiency?",
            None,
            "realistic_perfectionism_workplace"
        ),
        
        # Coping with a parent's remarriage
        (
            "My father remarried last year to someone my age, and I'm having a hard time adjusting. I'm 30 and have to call my dad's new wife 'step-mom' which feels ridiculous. She's trying to be my friend and include me in their activities, but I just need time to adjust. I don't want to hurt my father's feelings, but I'm not ready for this relationship. How do I handle this delicately?",
            None,
            "realistic_parent_remarrriage_adjustment"
        ),
        
        # Managing bipolar disorder symptoms
        (
            "I have bipolar disorder and I'm in a stable relationship, but I worry about how it affects my partner and our future. During manic periods, I make impulsive decisions that cause problems. During depressive episodes, I'm emotionally unavailable. I want to protect my partner from the harder parts of my condition. How do I maintain a healthy relationship?",
            None,
            "realistic_bipolar_relationship_management"
        ),
        
        # Dealing with career change after layoff
        (
            "I was laid off from my marketing job of 12 years and I'm realizing I don't want to go back to that field. The industry has changed significantly, and I've discovered I enjoyed the training I did more than the marketing. I'm 40 and wondering if it's realistic to become a corporate trainer. Is it too late to pivot my career?",
            None,
            "realistic_career_pivot_after_layoff"
        ),
        
        # Navigating remote work relationships
        (
            "I work remotely and I'm concerned about my visibility and career progression. My in-office colleagues seem to get more recognition and opportunities. I struggle to maintain relationships with coworkers through video calls only. I'm wondering if remote work is holding my career back. How can I maintain professional relationships when working from home?",
            None,
            "realistic_remote_work_relationships"
        ),
        
        # Dealing with body image after weight gain
        (
            "I've gained 30 pounds over the last year due to stress and medication for depression. I'm struggling with how I look and feel in clothes I used to love. I'm working on my mental health, but it's affecting my confidence at work and in my marriage. I want to be healthy but I don't want to become obsessed with weight again. How do I balance self-care with acceptance?",
            None,
            "realistic_body_image_weight_gain"
        ),
        
        # Managing ADHD as an adult
        (
            "I was diagnosed with ADHD at 34 and it explains so much about my life. I've always struggled with organization, time management, and follow-through. I'm learning new strategies and taking medication, but I'm also grieving the person I thought I was. I feel like I'm behind others my age. How do I catch up and develop good habits?",
            None,
            "realistic_adhd_adult_diagnosis"
        ),
        
        # Coping with a cancer diagnosis
        (
            "I was diagnosed with breast cancer at 36 and I'm in treatment. I'm trying to maintain normalcy for my 6-year-old daughter while dealing with the physical and emotional effects. I'm concerned about how my illness affects her development. I'm also worried about my career and finances during treatment. How do I balance everything while focusing on my health?",
            None,
            "realistic_cancer_diagnosis_parenting"
        ),
        
        # Dealing with family financial burden
        (
            "My parents are in serious debt and are asking me for financial help. I have my own family to support and I'm not wealthy. My siblings aren't contributing, and I feel guilty saying no, but helping them would put my own family at risk. My parents say I'm the 'successful' one so I should help. How do I set boundaries?",
            None,
            "realistic_family_financial_boundaries"
        ),
        
        # Navigating workplace harassment report aftermath
        (
            "I reported sexual harassment at work and the process has been traumatic. The harasser was a senior colleague and the company's response felt inadequate. Now I feel uncomfortable at work, my colleagues seem to whisper about the situation, and I'm considering leaving. I reported because it was the right thing to do, but I feel like I'm being punished. How do I move forward?",
            None,
            "realistic_workplace_harassment_aftermath"
        ),
        
        # Dealing with a learning disability as adult
        (
            "I was diagnosed with dyslexia at 31 after struggling with reading and writing my whole life. It explains my academic challenges and why I've always avoided certain tasks. I'm learning strategies to manage it, but I feel like I'm behind in my career. How do I adapt my work processes and disclose my disability to my employer?",
            None,
            "realistic_learning_disability_adult_diagnosis"
        ),
        
        # Coping with social anxiety in professional settings
        (
            "I have social anxiety that's particularly acute in professional settings. I avoid networking events, struggle with speaking up in meetings, and turn down opportunities that require social interaction. It's affecting my career advancement and I feel like I'm hiding my potential. How do I overcome this while managing my anxiety?",
            None,
            "realistic_social_anxiety_professional"
        ),
        
        # Managing a long-distance relationship with pets
        (
            "My partner and I have been in a long-distance relationship for 2 years. I recently adopted a dog, which was a major decision. Now I'm torn between the dog I love and the relationship. The dog can't travel, and I can't easily visit my partner. I feel responsible for the dog but also want to be with my partner. How do I balance these commitments?",
            None,
            "realistic_ldr_pets_commitment"
        ),
        
        # Dealing with family addiction
        (
            "I'm an adult child of an alcoholic and I realize my relationships patterns are affected by my childhood. I attract emotionally unavailable people and struggle with trust. I'm in therapy working on this, but I still find myself trying to 'fix' others. How do I break these patterns and form healthy relationships?",
            None,
            "realistic_acoa_relationship_patterns"
        ),
        
        # Navigating cultural differences in marriage
        (
            "I'm married to someone from a different culture, and I'm struggling with different expectations around family involvement, gender roles, and traditions. My husband's family expects more involvement in our life than I'm comfortable with. I want to respect his culture while maintaining my own identity and boundaries. How do we find middle ground?",
            None,
            "realistic_intercultural_marriage"
        ),
        
        # Dealing with age-related body changes
        (
            "I'm 45 and noticing significant changes in my body that are affecting my confidence and sexuality. I feel like I'm losing my identity as an attractive person. My husband is also aging but doesn't seem affected in the same way. I'm struggling with how to feel comfortable in my skin again and communicate my feelings to my partner.",
            None,
            "realistic_age_body_changes"
        ),
        
        # Managing work stress during pregnancy
        (
            "I'm in my first trimester and trying to manage a high-stress job while dealing with morning sickness and fatigue. I'm not ready to share the pregnancy news at work because I don't want to be treated differently or seen as less committed. I'm also worried about job security and benefits during parental leave. How do I balance career and pregnancy?",
            None,
            "realistic_pregnancy_work_stress"
        ),
        
        # Coping with a child's behavioral issues
        (
            "My 8-year-old son has been diagnosed with oppositional defiant disorder and ADHD. His behavior is challenging at home and school. I feel judged by other parents and teachers who think I'm not disciplining him properly. I'm exhausted from constant conflicts and I'm worried about his future. How do I manage his behavior while maintaining our relationship?",
            None,
            "realistic_child_behavioral_issues"
        ),
        
        # Dealing with a career setback
        (
            "I was passed over for a promotion that I was confident I would get, and it went to a less qualified colleague who is friends with our manager. I feel like my contributions are being overlooked and I'm questioning whether I should look for work elsewhere. I'm also second-guessing my capabilities. How do I recover from this professional disappointment?",
            None,
            "realistic_career_setback_recovery"
        ),
        
        # Navigating a blended family as step-parent
        (
            "I'm a step-parent to two teenagers and struggling with how to be supportive while respecting the boundaries of the biological parent. The kids are resistant to me and I feel like I'm walking on eggshells. My partner wants me to be more involved, but I don't want to overstep. How do I build relationships in this complex family dynamic?",
            None,
            "realistic_step_parent_blended_family"
        ),
        
        # Dealing with financial infidelity
        (
            "I recently discovered that my spouse has been hiding credit card debt and online shopping purchases for months. I thought we were financially aligned, but this has made me question everything about our relationship. I'm angry, hurt, and worried about our financial future. How do I address this breach of trust and prevent it from happening again?",
            None,
            "realistic_financial_infidelity"
        ),
        
        # Managing expectations after success
        (
            "I recently achieved a major career milestone I've worked for 10 years, but instead of feeling satisfied, I feel more pressure and imposter syndrome. Now everyone expects a lot from me, and I feel like I need to constantly prove myself. The success feels like more of a burden than an achievement. How do I handle success without losing myself?",
            None,
            "realistic_success_pressure"
        ),
        
        # Coping with a chronic illness affecting career
        (
            "I have Crohn's disease that requires frequent doctor visits and causes unpredictable flare-ups. I've had to modify my work schedule and use accommodations, but I feel like I'm not meeting expectations. I'm worried about being seen as unreliable and missing advancement opportunities. How do I manage a demanding career with an unpredictable chronic illness?",
            None,
            "realistic_chronic_illness_career_impact"
        ),
        
        # Dealing with family disownment over life choice
        (
            "My family disowned me after I chose to become a Buddhist and follow a minimalist lifestyle, which goes against our traditional family values. I lost not only my family but also financial support for my ongoing education. I'm on my own now, financially and emotionally. How do I build a life and support system from scratch?",
            None,
            "realistic_family_disownment_life_choice"
        ),
        
        # Navigating workplace politics as introvert
        (
            "I'm an introvert in a highly political workplace where relationship-building seems essential for advancement. I excel at my technical work but struggle with the social aspects of networking and self-promotion. I see extroverted colleagues getting opportunities that I'm qualified for. How do I navigate workplace politics while staying true to my personality?",
            None,
            "realistic_introvert_workplace_politics"
        ),
        
        # Dealing with religious deconversion
        (
            "I've quietly deconverted from Christianity after 30 years and I'm struggling with the loss of identity and community. I still love my religious family but I don't know how to be honest about my beliefs without causing conflict. I feel like I'm living a double life. How do I maintain relationships while being authentic?",
            None,
            "realistic_religious_deconversion"
        ),
        
        # Managing a career pivot during economic uncertainty
        (
            "I want to leave my corporate job to start a sustainable fashion business, but the economic climate feels risky. I have a family to support and a mortgage. I've saved 6 months of expenses but I'm worried about market volatility and supply chain issues. When is the right time to take a risk, especially when the external conditions seem uncertain?",
            None,
            "realistic_career_pivot_economic_uncertainty"
        ),
        
        # Coping with a parent with dementia
        (
            "My mother has early-stage dementia and I'm trying to balance respecting her autonomy with keeping her safe. She forgets to pay bills, gets lost driving, and makes questionable financial decisions. She doesn't want to admit she needs help and gets upset when I try to intervene. How do I support aging parents while respecting their dignity?",
            None,
            "realistic_parent_dementia_care"
        ),
        
        # Dealing with a midlife crisis
        (
            "I'm 48 and questioning every decision I've made. I have a stable career, good marriage, and grown children, but I feel like I've lost myself in meeting other people's expectations. I'm considering ending my marriage, changing careers, and moving to start over. Is this normal? How do I find myself without destroying everything?",
            None,
            "realistic_midlife_crisis"
        ),
        
        # Managing remote team relationships
        (
            "I'm a team manager with all-remote employees and I'm struggling to build team cohesion and maintain accountability. Some team members seem disengaged and I'm not sure if it's performance issues or personal problems. I've never managed remotely before and I'm concerned about fairness and team health. How do I maintain team effectiveness without micromanaging?",
            None,
            "realistic_remote_team_management"
        ),
        
        # Dealing with a toxic friendship
        (
            "I have a friend who brings me down more than she builds me up. She constantly complains, gossips about mutual friends, and makes everything about herself. I've been supporting her through her problems for years, but she's not there for me. I'm tired of being emotionally drained. How do I end a friendship gracefully?",
            None,
            "realistic_toxic_friendship"
        ),
        
        # Navigating a career gap due to mental health
        (
            "I took a year off work for mental health reasons and I'm ready to return, but I'm worried about the gap on my resume and explaining it in interviews. I feel like I'm behind my peers and that employers will see me as unreliable. I learned a lot during my time off, but I don't know how to frame it positively. How do I re-enter the job market?",
            None,
            "realistic_mental_health_career_gap"
        ),
        
        # Coping with empty nest as a single parent
        (
            "My daughter just left for college and I'm alone for the first time in 18 years. As a single parent, my entire identity was wrapped up in being a mom. I don't know who I am without her daily presence. I'm proud of her independence but devastated by the silence at home. How do I rediscover my individual identity?",
            None,
            "realistic_single_parent_empty_nest"
        ),
        
        # Dealing with age-related career concerns
        (
            "I'm 55 and see my younger colleagues getting opportunities that I used to get. I feel like my ideas are less valued and my experience is seen as being 'outdated.' I'm worried about my career trajectory and job security as I think about retirement. How do I stay relevant and valued as I age in my profession?",
            None,
            "realistic_age_related_career_concerns"
        ),
        
        # Managing a high-conflict divorce
        (
            "I'm going through a contentious divorce where my spouse is trying to turn our children against me through the courts. I'm spending more on lawyers than I make monthly and I'm fighting for custody. The stress is affecting my work and my health. I feel like I'm losing everything I worked for. How do I maintain my well-being during this battle?",
            None,
            "realistic_high_conflict_divorce"
        ),
        
        # Navigating career advancement as caregiver
        (
            "I'm a primary caregiver for an aging parent while trying to advance my career. I have to take time off for medical appointments and emergencies, which affects my availability for projects and overtime. I'm missing opportunities for advancement because of my caregiving responsibilities. How do I balance career goals with family obligations?",
            None,
            "realistic_career_advancement_caregiver"
        ),
        
        # Dealing with a family member's addiction
        (
            "My adult son is struggling with addiction and has been in and out of treatment. He promises to change but keeps falling back into old patterns. I love him but I'm struggling with how to help without enabling. My other children are affected by the family chaos. How do I support my son while protecting the family?",
            None,
            "realistic_adult_son_addiction"
        ),
        
        # Managing perfectionism in parenting
        (
            "I'm a new parent and I'm struggling with perfectionist expectations for myself and my baby. I feel like I'm constantly failing to meet the 'right' milestones, diet recommendations, and child-rearing practices. Social media makes me feel like everyone else is doing it better. How do I find balance in parenting?",
            None,
            "realistic_perfectionist_parenting"
        ),
        
        # Coping with job loss during family crisis
        (
            "I was laid off from my marketing job just as my father was diagnosed with cancer. I'm helping my mother care for him while job searching, and the stress is overwhelming. I'm not performing well in interviews because my mind is elsewhere. I need income to support my family during a medical crisis. How do I manage both responsibilities?",
            None,
            "realistic_job_loss_family_crisis"
        ),
        
        # Navigating relationship with in-laws
        (
            "My in-laws constantly undermine my decisions and my husband doesn't set boundaries with them. They give unsolicited advice about parenting, finances, and our lifestyle. I've tried being polite but they continue to interfere. My husband says they mean well and I should be more understanding. How do I protect my family without damaging relationships?",
            None,
            "realistic_in_law_boundaries"
        ),
        
        # Dealing with discrimination in healthcare
        (
            "I'm a Black woman experiencing discrimination in healthcare settings where my symptoms are dismissed or attributed to stress. I've had to advocate hard for proper diagnostic tests and treatment. It's exhausting to fight for proper care while I'm already unwell. How do I navigate a system that doesn't value my health?",
            None,
            "realistic_healthcare_discrimination"
        ),
        
        # Managing career during fertility treatments
        (
            "I'm going through IVF treatments which require time off and have physical side effects that affect my work performance. I'm not ready to tell my employer about my fertility journey, so I'm constantly juggling appointments and symptoms. I'm worried about career advancement while my energy is focused elsewhere. How do I manage both?",
            None,
            "realistic_fertility_treatments_career"
        ),
        
        # Coping with a major mistake at work
        (
            "I made an expensive error at work that cost the company $50k. I've owned up to it, but now I live in fear of making another mistake. My confidence is shot and I second-guess every decision. My manager says it's water under the bridge, but I can't stop thinking about the impact of my error. How do I move forward?",
            None,
            "realistic_workplace_mistake_recovery"
        ),
        
        # Dealing with anxiety about climate activism
        (
            "I'm passionate about climate change and I'm involved in activism, but it's causing anxiety and hopelessness. I'm angry at inaction from world leaders and corporations, but I feel powerless as an individual. My activism is affecting my sleep and my relationships with friends who don't share my urgency. How do I stay involved without burning out?",
            None,
            "realistic_climate_activism_anxiety"
        ),
        
        # Navigating a career change with family obligations
        (
            "I want to leave my corporate job to become a teacher, which would mean a significant pay cut, but I want a more meaningful career. I have two young children and a mortgage. My spouse supports me emotionally but we're concerned about practical implications. How do I balance passion with financial responsibility?",
            None,
            "realistic_career_change_family_obligations"
        ),
        
        # NEW CONVERSATIONAL TESTS - 50 diverse scenarios across various topics
        
        # Philosophy and existential questions
        (
            "I've been reading about existentialism and Camus' concept of absurdism lately, and it's making me question the meaning of my daily routines. I wake up, work, come home, repeat. Is this really all there is? I'm not depressed, just... wondering if I should be looking for something more profound. How do people find meaning in ordinary life?",
            None,
            "existential_meaning_search"
        ),
        
        # Food and cooking passion
        (
            "I've been cooking with my grandmother's recipes from Sicily and I want to start a food blog to preserve our family's culinary heritage. My cousins are scattered around the world and these recipes connect us. Should I focus on authenticity or adapt them for modern kitchens? I'm worried about commercializing something so personal.",
            None,
            "culinary_heritage_blog"
        ),
        
        # Music production and creativity
        (
            "I've been producing electronic music in my bedroom studio for 5 years. I just released my first track on Spotify and got 200 plays in a week, which feels amazing. Should I invest more money in better equipment or is my current setup good enough? I don't want to be an amateur forever but I also don't want to waste money.",
            None,
            "music_production_investment"
        ),
        
        # Gardening and sustainability
        (
            "I converted my entire backyard into a permaculture food forest and now I'm harvesting more vegetables than my family can eat. I want to share with neighbors but I don't want to seem preachy about sustainable living. How do I spread the joy of growing food without being annoying? Also, any tips for preserving excess produce?",
            None,
            "permaculture_sharing"
        ),
        
        # Language learning journey
        (
            "I've been learning Japanese for 2 years through anime and apps, and I can read hiragana and katakana fluently now. I want to visit Tokyo next year but I'm nervous about actually speaking. Should I hire a tutor for conversation practice or just dive in when I get there? I'm worried I'll freeze up.",
            None,
            "japanese_learning_conversation"
        ),
        
        # Woodworking and craftsmanship
        (
            "I built my first piece of furniture - a dining table from reclaimed barn wood. It took me 3 months of weekends and my wife loves it. Now friends are asking me to build stuff for them. Should I start a side business or keep it as a hobby? I have a full-time job but I love working with my hands.",
            None,
            "woodworking_side_business"
        ),
        
        # Astronomy and stargazing
        (
            "I bought my first telescope last month and I'm obsessed with astrophotography. I've captured images of Jupiter's moons and the Orion Nebula. My wife thinks I'm spending too much time outside at night but this hobby brings me so much peace. How do I balance my new passion with family time?",
            None,
            "astrophotography_passion"
        ),
        
        # Book club and literature discussion
        (
            "I run a book club with 8 friends and we've been reading classics for a year. We just finished '100 Years of Solitude' and had the most amazing discussion about magical realism and Latin American history. What should we read next that's similarly rich and discussion-worthy? We want to be challenged.",
            None,
            "book_club_recommendation"
        ),
        
        # Film analysis and cinema
        (
            "I've been studying Tarkovsky's films and I'm blown away by his use of long takes and philosophical themes. I want to understand more about Soviet cinema and the constraints directors worked under. Can you recommend other directors from that era who worked with similar poetic visual language?",
            None,
            "soviet_cinema_study"
        ),
        
        # Meditation and mindfulness practice
        (
            "I've been practicing Vipassana meditation daily for 6 months and it's changed my relationship with anxiety and stress. I went to a 10-day silent retreat last month. Now I'm thinking about becoming a meditation teacher, but I feel like 6 months isn't enough experience. How long should I practice before teaching others?",
            None,
            "meditation_teaching_timing"
        ),
        
        # Urban exploration and photography
        (
            "I've been exploring abandoned buildings and industrial sites for photography. The decay and history fascinate me. I'm careful about safety and trespassing laws, but I'm wondering about the ethics of sharing these images. Am I exploiting urban decay for aesthetic purposes or documenting important history?",
            None,
            "urban_exploration_ethics"
        ),
        
        # Podcast creation journey
        (
            "My friend and I started a podcast about true crime in our local area, focusing on unsolved cases. We've released 5 episodes and have 500 downloads. We're passionate about it but wondering if we need better equipment or if we should focus on content quality first. What makes a podcast successful?",
            None,
            "true_crime_podcast_growth"
        ),
        
        # Martial arts training philosophy
        (
            "I've been training Brazilian Jiu-Jitsu for 3 years and just got my blue belt. The physical aspect is great but what I love most is the problem-solving and chess-like strategy. I'm thinking about competing but I'm 38 and worried about injuries. Should older practitioners compete or just train for personal growth?",
            None,
            "bjj_competition_age"
        ),
        
        # Vintage fashion and thrifting
        (
            "I've been thrifting and curating vintage clothing for years, and my apartment is basically a museum of 1960s-1980s fashion. I document my finds on Instagram and people keep asking where to buy. Should I start selling or does that change the joy of the hunt? I love the sustainability aspect.",
            None,
            "vintage_fashion_selling"
        ),
        
        # Gaming and esports discussion
        (
            "I've been playing competitive Valorant and I'm ranked in the top 2% of players. My team wants to try competing in small tournaments but I'm worried about the time commitment. I have a full-time job and gaming is my way to decompress. How do semi-professional gamers balance competition with real life?",
            None,
            "esports_time_balance"
        ),
        
        # Baking science and experimentation
        (
            "I've become obsessed with the chemistry of bread baking. I maintain three sourdough starters and I've been experimenting with different flours, hydration levels, and fermentation times. My coworkers love when I bring in fresh bread. What's your understanding of how gluten development affects crumb structure?",
            None,
            "sourdough_science_exploration"
        ),
        
        # Architecture appreciation and history
        (
            "I've been taking walking tours to study Brutalist architecture in my city. Everyone seems to hate these buildings but I find them fascinating - the raw concrete, the bold geometric forms. I'm documenting them before they're demolished. How do I help people appreciate architectural styles that are considered ugly?",
            None,
            "brutalist_architecture_appreciation"
        ),
        
        # Comic book collecting and analysis
        (
            "I've been collecting indie comics and graphic novels, focusing on non-superhero narratives. I'm particularly interested in how sequential art can tell stories that prose can't. I'm thinking about writing a blog analyzing narrative techniques in graphic storytelling. What are some groundbreaking examples I should study?",
            None,
            "graphic_novel_narrative_study"
        ),
        
        # Cycling and bike touring
        (
            "I'm planning a solo bike tour from Seattle to San Francisco next summer. I've done weekend trips but nothing this ambitious. I'm excited but also nervous about bike mechanicals, finding camping spots, and being alone for that long. What should I know that I don't know to ask about long-distance touring?",
            None,
            "bike_touring_preparation"
        ),
        
        # Home brewing and fermentation
        (
            "I started home brewing beer 18 months ago and now I'm experimenting with sour ales and wild fermentation. My last batch turned out amazing - complex, funky, perfectly balanced. Friends say I should sell it but the regulations seem complicated. Is home brewing better kept as a passion or can it become a business?",
            None,
            "home_brewing_business_consideration"
        ),
        
        # Genealogy research journey
        (
            "I've been researching my family history and I traced my ancestors back to 1820s Ireland. I found immigration records, census data, and I'm connecting with distant cousins. This has given me such a sense of identity and belonging. How far back can realistically trace family history and what are the best resources?",
            None,
            "genealogy_research_depth"
        ),
        
        # Bird watching and conservation
        (
            "I got into birding during lockdown and it's become my favorite weekend activity. I've logged 150 species in my area using eBird. I'm particularly concerned about habitat loss affecting migratory patterns. How can citizen science like bird counts actually contribute to conservation efforts?",
            None,
            "birding_conservation_impact"
        ),
        
        # Chess improvement journey
        (
            "I've been playing chess seriously for a year, working through tactics puzzles daily and watching instructional videos. My rating has plateaued around 1400 and I'm frustrated. I know the opening principles but I struggle in the middlegame. What's the best way to break through this plateau?",
            None,
            "chess_rating_plateau"
        ),
        
        # Calligraphy and lettering art
        (
            "I've been practicing modern calligraphy and hand lettering for my wedding invitations. It's meditative and I love how analog it is in our digital world. People are asking me to do invitations for their events. Should I charge money or keep it as a hobby? I don't want to ruin something I love.",
            None,
            "calligraphy_monetization"
        ),
        
        # Rock climbing and outdoor adventure
        (
            "I progressed from gym climbing to outdoor sport climbing last year. I just led my first 5.11a route and I'm hooked. Now I want to learn trad climbing but it's intimidating - placing your own gear, more complex anchor systems. How did you make the transition from sport to trad?",
            None,
            "trad_climbing_transition"
        ),
        
        # Fashion design and sewing
        (
            "I taught myself to sew during the pandemic and now I'm making all my own clothes. I love the fit and the uniqueness. I'm thinking about studying pattern making more seriously. Is formal fashion education necessary or can self-taught sewists reach professional level?",
            None,
            "fashion_sewing_education"
        ),
        
        # Marine biology interest
        (
            "I'm fascinated by coral reef ecosystems and I've been reading research papers about coral bleaching and restoration efforts. I have a degree in business but I'm considering going back to school for marine biology. I'm 31 - is it too late for such a dramatic career pivot?",
            None,
            "marine_biology_career_change"
        ),
        
        # Stand-up comedy pursuit
        (
            "I've been doing open mics for 6 months, working on my 5-minute set. Comedy is terrifying but exhilarating. I bombed hard twice but I've also had sets where I felt the audience connection. How do comedians develop their unique voice and know when they're ready for paid gigs?",
            None,
            "standup_comedy_development"
        ),
        
        # Historical reenactment hobby
        (
            "I joined a Viking Age historical reenactment group and I'm learning blacksmithing, tablet weaving, and studying Old Norse. It's incredible to connect with history through hands-on crafts. Some people think it's weird but I find it enriching. How do you explain immersive historical hobbies to people who don't get it?",
            None,
            "historical_reenactment_passion"
        ),
        
        # Magic and sleight of hand
        (
            "I've been practicing card magic and sleight of hand for 2 years. I perform for friends and family and they're amazed. I love the psychology and storytelling aspects as much as the technical skills. Should I pursue this semi-professionally or will monetizing it remove the magic?",
            None,
            "card_magic_performance"
        ),
        
        # Mycology and mushroom foraging
        (
            "I've become obsessed with mushroom identification and foraging. I've been studying field guides, joining foray groups, and I successfully identified and harvested chanterelles last fall. The ecological role of fungi is mind-blowing. What are the best resources for learning mushroom identification safely?",
            None,
            "mushroom_foraging_learning"
        ),
        
        # Animation and motion graphics
        (
            "I've been teaching myself 2D animation using Procreate and After Effects. I want to create animated explainer videos that make complex topics accessible. Should I build a portfolio and look for freelance work or keep developing my skills first? When do you know you're 'good enough'?",
            None,
            "animation_portfolio_timing"
        ),
        
        # Vintage car restoration
        (
            "I bought a 1972 Volkswagen Beetle that needs a complete restoration. I have basic mechanical knowledge but this is my first major project. I'm documenting the process on YouTube. The community has been incredibly helpful. How do you stay motivated during a multi-year restoration project?",
            None,
            "car_restoration_motivation"
        ),
        
        # Typography and design theory
        (
            "I've fallen down a rabbit hole studying typography and typeface design. I'm amazed by how subtle changes in letterforms affect readability and emotion. I've been practicing lettering and I want to design my own font. What's the learning path from appreciation to creation?",
            None,
            "typography_design_path"
        ),
        
        # Beekeeping and apiculture
        (
            "I started beekeeping last year with two hives and I'm fascinated by colony behavior and honey production. My first harvest was incredible - 40 pounds of honey. I'm concerned about colony collapse and what I can do at a local level. How impactful is small-scale beekeeping for pollinator conservation?",
            None,
            "beekeeping_conservation_impact"
        ),
        
        # Improv theater and performance
        (
            "I joined an improv theater group 8 months ago and it's pushing me out of my comfort zone in the best way. I'm learning to trust my instincts and build on others' ideas. It's made me better at public speaking and collaboration at work. How does improv training transfer to other areas of life?",
            None,
            "improv_life_skills"
        ),
        
        # Pottery and ceramics
        (
            "I've been taking pottery classes for a year and finally threw my first successful bowl on the wheel. The centering and pulling up walls is so difficult but when it clicks, it's magical. I love the tactile nature of working with clay. Should I invest in a home kiln or stick with studio access?",
            None,
            "pottery_equipment_investment"
        ),
        
        # Geocaching and outdoor puzzles
        (
            "I discovered geocaching and I'm hooked on the treasure hunt aspect. I've found 50 caches so far, from simple containers to elaborate puzzles. I love that it gets me exploring areas I'd never normally visit. What are the most creative or challenging geocaches you've encountered?",
            None,
            "geocaching_experiences"
        ),
        
        # Whiskey tasting and appreciation
        (
            "I've been learning about whiskey tasting and distillation processes. I've tried 30 different single malts and I'm starting to identify flavor profiles and regional characteristics. I'm thinking about visiting distilleries in Scotland. How did you develop your palate for appreciating complex spirits?",
            None,
            "whiskey_appreciation_learning"
        ),
        
        # Writing fiction and storytelling
        (
            "I've been writing short stories for a year and I just got my first rejection from a literary magazine. I know rejection is part of the process but it still stings. How many rejections should I expect before getting published? And how do you maintain motivation when the validation is so rare?",
            None,
            "fiction_writing_rejection"
        ),
        
        # Scuba diving and underwater exploration
        (
            "I got my scuba certification last year and completed 15 dives so far. The underwater world is unlike anything I've experienced. I'm planning a liveaboard trip to the Great Barrier Reef. What should I know as a relatively new diver about advanced dive planning and safety?",
            None,
            "scuba_diving_advancement"
        ),
        
        # Mathematical puzzles and problem-solving
        (
            "I've been working through Project Euler problems and I love the intersection of math and programming. I can solve the early problems but the difficulty ramps up quickly. How do mathematicians develop the intuition for these types of problems? Is it just practice or are there fundamental concepts to master?",
            None,
            "project_euler_problem_solving"
        ),
        
        # Leather working and craftsmanship
        (
            "I started making leather goods - wallets, belts, bags - using traditional hand-stitching techniques. The craftsmanship and durability appeal to me. I've been selling a few pieces to friends and at local markets. How do I price handmade goods so I'm fairly compensated for my time?",
            None,
            "leather_crafting_pricing"
        ),
        
        # Parkour and movement training
        (
            "I've been training parkour for 18 months and I love the creative problem-solving of movement. It's changed how I see urban environments. I'm 29 and sometimes I feel too old to be learning these skills, but the community is welcoming. What are realistic progression goals for adult beginners?",
            None,
            "parkour_adult_progression"
        ),
        
        # Telescope making and astronomy
        (
            "I'm attempting to grind my own telescope mirror from scratch using the classic techniques. It's a months-long process of grinding and polishing glass. The precision required is incredible. Have you ever built optical instruments? What's the most challenging DIY optics project you've done?",
            None,
            "telescope_making_project"
        ),
        
        # Herbalism and plant medicine
        (
            "I've been studying herbalism and growing medicinal plants in my garden. I'm making tinctures, salves, and teas from plants like calendula, echinacea, and chamomile. I'm fascinated by traditional plant knowledge. How do you balance traditional herbal medicine with modern medical understanding?",
            None,
            "herbalism_modern_balance"
        ),
        
        # Swimming technique improvement
        (
            "I've been swimming for fitness for years but recently started working on proper technique. I hired a coach and I'm amazed at how much more efficient swimming can be with proper form. I'm training for my first open water swim event. What was the breakthrough that took your swimming to the next level?",
            None,
            "swimming_technique_breakthrough"
        ),
        
        # Origami and paper folding
        (
            "I've been folding origami for 3 years and progressing to complex modular designs. I just completed a kusudama with 30 individual units. The mathematical precision and geometric beauty fascinate me. I'm thinking about designing my own original models. How do origami designers approach creating new folds?",
            None,
            "origami_design_process"
        ),
        
        # Espresso and coffee roasting
        (
            "I've gone deep into espresso making - I have a proper machine, grinder, and I'm dialing in shots by adjusting grind size, temperature, and pressure. Now I want to start roasting my own beans. Is home roasting worth the investment or should I stick with buying from quality roasters?",
            None,
            "coffee_roasting_investment"
        ),
        
        # Dog training and behavior
        (
            "I rescued a reactive dog and I've been working with a trainer using positive reinforcement methods. The progress has been slow but amazing to see. I'm learning so much about animal behavior and communication. I'm considering getting certified as a dog trainer myself. Is this career realistic for someone in their mid-30s?",
            None,
            "dog_training_career_transition"
        ),
    ]
    
    # Edge Cases and Tricky Tests
    EDGE_CASE_TESTS = [
        # Should be conversational (personal context)
        (
            "I'm learning Python to automate tasks at my job",
            None,
            "edge_personal_tech"
        ),
        (
            "Can you help me understand quantum computing for my physics course?",
            None,
            "edge_personal_learning"
        ),
        
        # Borderline cases
        (
            "I'm writing a story for my creative writing class about my childhood",
            None,
            "edge_personal_fiction"
        ),
        
        # ADDITIONAL COMPLEX EDGE CASES
        
        # Mixed Technical and Personal Context
        (
            """I'm building a React Native app for my mother who has arthritis. She struggles with small touch targets on her phone, so I'm implementing larger buttons and voice controls. 
            
I'm using React Native Voice library for speech recognition and AsyncStorage for saving her preferences. The tricky part is making the voice commands work reliably - she has a strong Southern accent and the default recognition doesn't handle it well.

I've been training a custom model, but I'm not sure if that's overkill for a personal project. Should I just adjust the confidence thresholds? Or maybe use a different speech recognition service?

This is really important to me because I want her to be able to use technology independently. She gets frustrated when she can't tap accurately and it breaks my heart.""",
            None,
            "edge_technical_personal_motivation"
        ),
        
        # Work Problem with Deep Personal Stakes
        (
            """My team at work is about to make a decision that I think is ethically wrong, and I don't know what to do.
            
Context: I'm a data scientist at a health insurance company. We've been asked to build a predictive model to identify members who are likely to file expensive claims in the next year. The goal is to "proactively offer wellness programs" - but I've seen the internal emails, and it's clear they want to use this to push people toward cheaper plans or find reasons to audit claims more closely.

The model I built works really well - 87% accuracy. But the features it relies on most heavily are things like age, zip code, and number of previous claims. When I ran a bias audit, it's clearly discriminating against elderly people and those in low-income neighborhoods. These are the people who most need good insurance coverage.

My manager says "that's not our problem, we're just building what they asked for." The product team says the wellness programs are genuine. But I know how these things work in corporate environments - the tool will be used for cost-cutting, not helping people.

I have student loans. I need this job. But I also can't sleep at night thinking about the grandmother in rural Kentucky who might lose her coverage because of an algorithm I wrote.

Do I blow the whistle? Push back harder internally? Just implement it and move on? I feel like there's no good option here.""",
            None,
            "edge_work_ethical_dilemma"
        ),
        
        # Deeply Technical Problem with Emotional Context
        (
            """My open source project just got its first external contributor, and they submitted a pull request that completely breaks the architecture I've been carefully building for 2 years. But they're so enthusiastic and I don't want to crush their spirit.

The PR replaces my clean dependency injection system with a global singleton pattern. They added a God class that handles database, caching, logging, and API calls all in one 800-line file. There are no tests. The commit messages are things like "fixed it" and "updates."

But here's the thing - they wrote in the PR description that they've been learning to code for 6 months, this is their first open source contribution, and they're incredibly proud of this work. They told their family about it. They're already talking about what feature they want to work on next.

I remember being a beginner. I remember submitting terrible code and having maintainers either ignore me or be dismissive. It hurt and almost made me quit programming.

How do I reject this PR without being discouraging? I want to give constructive feedback, but I also need to protect the project. And honestly, reviewing and explaining all the issues is going to take hours I don't have.

The project has 2.3k stars. Other users are watching. I need to handle this right, but I'm exhausted from my day job and just want to code, not mentor.

What's the right balance between being kind and maintaining standards?""",
            None,
            "edge_technical_leadership_challenge"
        ),
        
        # Code Review vs Personal Conflict
        (
            """There's a senior developer on my team who keeps rejecting my PRs for reasons that feel personal rather than technical.

Today's example: I refactored a component to use React hooks instead of class components. It's cleaner, more testable, and follows our team's stated direction of moving to functional components. All tests pass. No breaking changes.

His review comment: "This is unnecessary churn. The class component worked fine. Rejected."

But last month, he refactored someone else's component from class to hooks and said it was "modernizing the codebase."

This happens constantly. If I suggest using TypeScript for a new feature: "Adds complexity." When he does it: "Type safety is important."

I use optional chaining: "Too clever, hard to read." He uses it: crickets.

I've tried everything:
- Being extra thorough in PR descriptions
- Following every code style guide exactly  
- Adding extensive tests
- Asking clarifying questions before starting work

Nothing helps. My PRs sit in review for days, then get rejected or nitpicked to death. Meanwhile, other team members' PRs get approved in hours.

I talked to my manager. He said "maybe your code quality needs improvement" - even though my 1:1 feedback has always been positive and I've never missed a deadline.

I think the real issue is that I questioned his technical decision in a meeting once (respectfully! With data!) and he's been holding a grudge for 4 months.

I'm spending more time dealing with his reviews than actually coding. It's affecting my performance metrics. And it's making me dread coming to work.

Is this normal? Am I being too sensitive? Do I escalate higher up? Start looking for a new job?""",
            None,
            "edge_workplace_conflict_technical"
        ),
        
        # Tutorial Request with Personal Context
        (
            """I need to learn Docker and Kubernetes fast because I lied on my resume and now I'm in over my head.

I know, I know - I shouldn't have lied. But I was desperate. I'd been unemployed for 8 months after getting laid off. My savings were gone. I was about to lose my apartment. When I saw this job posting that was perfect except for the K8s requirement, I thought "how hard can it be?"

Now it's day 3 and they want me to help migrate our application to Kubernetes. I've been nodding along in meetings, frantically Googling terms under the table. I stay late watching YouTube tutorials. I've learned what pods and deployments are, but I don't understand how it all fits together.

Tomorrow I have to pair program with the senior DevOps engineer to "review our current setup." I'm terrified he'll realize I have no idea what I'm doing.

Can you give me a crash course? Specifically:
- What are the absolute must-know concepts for someone who needs to sound competent?
- What are the common gotchas that would immediately reveal me as a fraud?
- Are there any good lies I can tell to buy more time? ("Oh, I used K8s but only in AWS EKS" or "I only used Docker Compose, not full orchestration"?)

I'm not a bad developer - I'm genuinely good at backend engineering. But I'm in serious trouble here. If I get fired, I won't get another chance. The tech market is brutal right now.

Please help me not ruin my life with this stupid decision.""",
            None,
            "edge_learning_desperation_context"
        ),
        
        # Code vs Relationship Balance
        (
            """My girlfriend is upset because I spent our anniversary weekend debugging a production issue instead of going on the trip we planned. I'm trying to explain why it was necessary, but she says I always prioritize work over her.

Here's what happened: Saturday morning, 6 AM, I get paged. Our payment processing system is down. We're losing $10,000 per hour. I'm the only one who knows this part of the codebase because the original developer left and I inherited it.

I told my girlfriend I'd try to fix it quickly. "Just give me an hour." 

It wasn't an hour. The bug was a race condition that only appeared under high load. I had to dive into the database transaction logs, add instrumentation, deploy to staging, reproduce the issue, and then implement a fix with proper locking.

By the time I got it working and deployed safely, it was 8 PM. Our hotel reservation was gone (non-refundable). The restaurant booking was past. The weekend was over.

She says: "You could have delegated it. You could have told your boss you were unavailable. You chose work."

I say: "I'm responsible. We would have lost hundreds of thousands of dollars. People depend on me."

But she's right that this keeps happening. I missed her birthday dinner for an on-call issue. I've canceled three vacations. I check Slack during movies. I bring my laptop everywhere "just in case."

I love my job. I love building things and solving problems. The rush of fixing a critical bug is... honestly more satisfying than most other things in my life. That's horrible to admit, but it's true.

But I also love her. We've been together 4 years. She's been patient and supportive. And I'm watching that fade into resentment.

How do other engineers handle this? Is it possible to be excellent at this career and also maintain a relationship? Or do I need to choose?

My dad was a workaholic. He was successful but divorced twice. I swore I wouldn't be like him. But here I am.""",
            None,
            "edge_work_life_relationship_crisis"
        ),
    ]


class SkipDetectorTester:
    """Comprehensive tester for SkipDetector with layer-by-layer analysis."""
    
    def __init__(self):
        print("Initializing SkipDetector with embedding model...", flush=True)
        self.embedding_model = SentenceTransformer(
            Constants.DEFAULT_EMBEDDING_MODEL, 
            device="cpu", 
            trust_remote_code=True
        )
        self.skip_detector = SkipDetector(self.embedding_model)
        
        self.results = {
            'size_layer': {'correct': 0, 'incorrect': 0, 'total': 0, 'details': []},
            'fast_path_layer': {'correct': 0, 'incorrect': 0, 'total': 0, 'details': []},
            'semantic_layer': {'correct': 0, 'incorrect': 0, 'total': 0, 'details': []},
            'overall': {'correct': 0, 'incorrect': 0, 'total': 0}
        }
    
    def _determine_layer(self, message: str, result: str) -> str:
        """Determine which layer caught the skip."""
        # Check size layer first
        size_result = self.skip_detector.validate_message_size(message, Constants.MAX_MESSAGE_CHARS)
        if size_result and result == size_result:
            return 'size_layer'
        
        # Check fast-path layer
        fast_result = self.skip_detector._fast_path_skip_detection(message)
        if fast_result and result == fast_result:
            return 'fast_path_layer'
        
        # Otherwise it's semantic layer (or no skip)
        return 'semantic_layer'
    
    def test_category(self, test_data: Sequence[Tuple[str, Optional[str], str]], category_name: str):
        """Test a specific category and track which layer handles it."""
        failing_tests = []
        
        for message, expected, test_id in test_data:
            result = self.skip_detector.detect_skip_reason(message)
            is_correct = result == expected
            
            # Determine which layer handled this
            layer = self._determine_layer(message, result if result else "None")
            
            # Update layer-specific stats
            self.results[layer]['total'] += 1
            if is_correct:
                self.results[layer]['correct'] += 1
            else:
                self.results[layer]['incorrect'] += 1
            
            # Store details
            detail = {
                'test_id': test_id,
                'message': message[:100] + '...' if len(message) > 100 else message,
                'expected': expected,
                'actual': result,
                'correct': is_correct,
                'layer': layer
            }
            self.results[layer]['details'].append(detail)
            
            # Collect failing tests
            if not is_correct:
                failing_tests.append((test_id, expected, result))
        
        # Print failing tests
        if failing_tests:
            for test_id, expected, actual in failing_tests:
                print(f"FAIL: {test_id} | Expected: {expected} | Got: {actual}", flush=True)
    
    def run_all_tests(self):
        """Run all test categories."""
        # Test each category
        self.test_category(TestDataset.SIZE_TESTS, "Size Validation Tests")
        self.test_category(TestDataset.FAST_PATH_TESTS, "Fast-Path Detection Tests")
        self.test_category(TestDataset.SEMANTIC_TESTS, "Semantic Classification Tests")
        self.test_category(TestDataset.CONVERSATIONAL_TESTS, "Conversational Tests (Should NOT Skip)")
        self.test_category(TestDataset.EDGE_CASE_TESTS, "Edge Case Tests")
        
        # Calculate overall stats
        for layer_name in ['size_layer', 'fast_path_layer', 'semantic_layer']:
            layer = self.results[layer_name]
            self.results['overall']['total'] += layer['total']
            self.results['overall']['correct'] += layer['correct']
            self.results['overall']['incorrect'] += layer['incorrect']
    
    def print_layer_analysis(self):
        """Print concise layer-by-layer analysis."""
        print("\nLAYER ANALYSIS", flush=True)
        print("-"*50, flush=True)
        
        for layer_name in ['size_layer', 'fast_path_layer', 'semantic_layer']:
            layer = self.results[layer_name]
            if layer['total'] == 0:
                continue
            
            accuracy = (layer['correct'] / layer['total'] * 100) if layer['total'] > 0 else 0
            print(f"{layer_name.replace('_', ' ').title():15s}: {layer['total']:3d} tests, {accuracy:.1f}% accuracy", flush=True)
    
    def print_final_report(self):
        """Print concise final report."""
        overall = self.results['overall']
        accuracy = (overall['correct'] / overall['total'] * 100) if overall['total'] > 0 else 0
        
        print(f"\nFINAL RESULTS: {overall['total']} tests, {accuracy:.1f}% accuracy", flush=True)
        print("="*50, flush=True)


def main():
    """Main test execution."""
    print("Script started...", flush=True)
    print("Starting SkipDetector comprehensive test suite...", flush=True)
    tester = SkipDetectorTester()
    print("SkipDetectorTester initialized successfully", flush=True)
    tester.run_all_tests()
    tester.print_layer_analysis()
    tester.print_final_report()


if __name__ == "__main__":
    print("Script started...", flush=True)
    main()
    print("Script completed.", flush=True)
