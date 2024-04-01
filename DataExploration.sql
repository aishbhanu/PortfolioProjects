Select *
from PortfolioProject..CovidDeaths
Where continent is not null
order by 3,4


-- Data being used
Select Location, date, total_cases, new_cases,total_deaths,population
from   PortfolioProject..CovidDeaths
order by 1,2  

-- Death percentage
-- likelihood  if you contract covid in your country
select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
from PortfolioProject..CovidDeaths
WHERE location like '%India%'
order by 1,2

--Total cases vs population
--Shows percentage of population with covid
select location, date, total_cases, population, (total_cases/population)*100 as PercentofPopultioninfected
from PortfolioProject..CovidDeaths
--WHERE location like '%India%'
order by 1,2

--Looking at countries with highest infection rate compared to Population
select location, population,MAX(total_cases) AS highestInfectionCount, MAX((total_cases/population))*100 
as PercentofPopultioninfected
from PortfolioProject..CovidDeaths
Group by location,population
--WHERE location like '%India%'
order by PercentofPopultioninfected desc


-- Breaking this down by continent
--Continents with higest death count per population
select continent, Max(cast(total_deaths as int)) as totalDeathCount
from PortfolioProject..CovidDeaths
Where continent is not null
Group by continent
order by totalDeathCount desc

-- Global numbers

Select  SUM(new_cases) as Total_cases,
SUM(cast(new_deaths as int))as Total_Deaths,
SUM(cast(new_deaths as int))/SUM(new_cases) *100 as DeathPerectange
from PortfolioProject..CovidDeaths
where continent is not null
--group by date
order by 1,2



-- Joining Tables covid_death and Covid_vaccination tables
Select * 
from PortfolioProject..CovidDeaths cd
join PortfolioProject..CovidVaccinations cv
on cd.location=cv.location
and cd.date=cv.date

-- Showing total population vs total vaccination
-- Use CTE

With PopvsVac ( continent, location,date, population, new_vaccinations, RollingPeopleVaccinated) 
as 
(
Select cd.continent, cd.location, cd.date, cd.population, cv.new_vaccinations,
SUM(CONVERT(int,cv.new_vaccinations)) OVER (Partition by cd.location order by cd.location, cd.date) as RollingPeopleVaccinated
from PortfolioProject..CovidDeaths cd
join PortfolioProject..CovidVaccinations cv
on cd.location=cv.location
and cd.date=cv.date
where cd.continent is not null
--order by 2,3
)

Select *, (RollingPeopleVaccinated/population)*100 
From PopvsVac



-- Using TEMP TABLE

DROP TABLE if exists #percentpopulationvaccinated -- inorder to make any updates
Create Table #percentpopulationvaccinated
(
Continent nvarchar(255),
Location varchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)

Insert into #percentpopulationvaccinated
Select cd.continent, cd.location, cd.date, cd.population,
cv.new_vaccinations,
SUM(CONVERT(int,cv.new_vaccinations)) 
OVER (Partition by cd.location order by cd.location, cd.date) as RollingPeopleVaccinated
from PortfolioProject..CovidDeaths cd
join PortfolioProject..CovidVaccinations cv
on cd.location=cv.location
and cd.date=cv.date
where cd.continent is not null
--order by 2,3

Select *, (RollingPeopleVaccinated/population)*100 
 From #percentpopulationvaccinated


 -- Creating a View
 
Create View percentofpopulationvaccinated as
 Select cd.continent, cd.location, cd.date, cd.population,
cv.new_vaccinations,
SUM(CONVERT(int,cv.new_vaccinations)) 
OVER (Partition by cd.location order by cd.location, cd.date) as RollingPeopleVaccinated
from PortfolioProject..CovidDeaths cd
join PortfolioProject..CovidVaccinations cv
on cd.location=cv.location
and cd.date=cv.date
where cd.continent is not null

Select * from percentofpopulationvaccinated
