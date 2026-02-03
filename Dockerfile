FROM dolfinx/dolfinx:stable

# Install uv
RUN pip install uv

# Copy project files
WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync