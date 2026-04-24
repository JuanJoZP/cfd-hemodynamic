FROM dolfinx/dolfinx:v0.9.0

# Install uv
RUN pip install uv

# Copy project files
WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync
