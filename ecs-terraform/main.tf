provider "aws" {
  region = "us-east-1"
}

resource "aws_ecs_cluster" "e2e_ml_cluster" {
  name = "e2e-ml-cluster"
}

resource "aws_vpc" "e2e_ml_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support = true
  enable_dns_hostnames = true
  tags = {
    Name = "e2e-ml-vpc"
  }
}

resource "aws_subnet" "e2e_ml_subnet" {
  vpc_id            = aws_vpc.e2e_ml_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"
  tags = {
    Name = "e2e-ml-subnet"
  }
}

resource "aws_security_group" "e2e_ml_sg" {
  name        = "e2e-ml-sg"
  description = "Allow traffic for e2e ML service"
  vpc_id      = aws_vpc.e2e_ml_vpc.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "e2e-ml-security-group"
  }
}


resource "aws_internet_gateway" "e2e_ml_igw" {
  vpc_id = aws_vpc.e2e_ml_vpc.id

  tags = {
    Name = "e2e-ml-igw"
  }
}

resource "aws_route_table" "e2e_ml_rt" {
  vpc_id = aws_vpc.e2e_ml_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.e2e_ml_igw.id
  }

  tags = {
    Name = "e2e-ml-rt"
  }
}

resource "aws_route_table_association" "e2e_ml_rta" {
  subnet_id      = aws_subnet.e2e_ml_subnet.id
  route_table_id = aws_route_table.e2e_ml_rt.id
}


resource "aws_ecs_task_definition" "e2e_ml_task" {
  family                   = "e2e-ml-task"
  cpu                      = "256"
  memory                   = "512"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.e2e_ml_execution_role.arn
  container_definitions = jsonencode([
    {
      name      = "e2e-ml-container"
      image     = "voynow/sentiment-analysis-api:latest"
      cpu       = 256
      memory    = 512
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/e2e-ml-service"
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "e2e_ml_service" {
  name            = "e2e-ml-service"
  cluster         = aws_ecs_cluster.e2e_ml_cluster.id
  task_definition = aws_ecs_task_definition.e2e_ml_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    assign_public_ip = true
    subnets         = [aws_subnet.e2e_ml_subnet.id]
    security_groups = [aws_security_group.e2e_ml_sg.id]
  }

  depends_on = [
    aws_ecs_task_definition.e2e_ml_task,
  ]
}

resource "aws_iam_role" "e2e_ml_execution_role" {
  name = "e2e_ml_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "e2e_ml_execution_role_policy" {
  role       = aws_iam_role.e2e_ml_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_cloudwatch_log_group" "ecs_logs" {
  name = "/ecs/e2e-ml-service"
  retention_in_days = 14
}
