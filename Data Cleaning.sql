/* Cleaning Data in SQL Queries*/


-- Standardising Date Formate
Select *
From PortfolioProject.dbo.[Nashville Housing]

Select SaleDate, CONVERT(Date,SaleDate)
From PortfolioProject.dbo.[Nashville Housing]

ALTER TABLE [Nashville Housing]
ADD SaleDate_Converted Date

Update [Nashville Housing]
SET SaleDate_Converted = CONVERT(Date, SaleDate)
---------------------------------------------------------------------------
-- Populate Property Address Data 

Select * 
FROM [Nashville Housing]
--Where PropertyAddress is null
order by ParcelID


Select nh1.ParcelID, nh1.PropertyAddress,nh2.ParcelID, nh2.PropertyAddress,ISNULL(nh1.PropertyAddress,nh2.PropertyAddress)
FROM [Nashville Housing] nh1
	JOIN  [Nashville Housing] nh2
	on nh1.ParcelID = nh2.ParcelID
	and nh1.[UniqueID ] <> nh2.[UniqueID ]
Where nh1.PropertyAddress is null

Update nh1
SET PropertyAddress = ISNULL(nh1.PropertyAddress,nh2.PropertyAddress)
FROM [Nashville Housing] nh1
	JOIN  [Nashville Housing] nh2
	on nh1.ParcelID = nh2.ParcelID
	and nh1.[UniqueID ] <> nh2.[UniqueID ]


---------------------------------------------------------------------
--Breaking Address into individual columns ( Address, City, State)

Select PropertyAddress
FROM [Nashville Housing]
--Where PropertyAddress is null
order by ParcelID

select 
SUBSTRING(PropertyAddress, 1,CHARINDEX(',',PropertyAddress)-1) AS Address
,SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,LEN(PropertyAddress)) AS City
FROM [Nashville Housing]

ALTER TABLE [Nashville Housing]
ADD PropertyStreetAddress nvarchar(255)

Update [Nashville Housing]
SET PropertyStreetAddress = SUBSTRING(PropertyAddress, 1,CHARINDEX(',',PropertyAddress)-1)

ALTER TABLE [Nashville Housing]
ADD PropertyCityAddress nvarchar(255)

Update [Nashville Housing]
SET PropertyCityAddress = SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,LEN(PropertyAddress))




Select OwnerAddress
from [Nashville Housing]

Select 
PARSENAME(REPLACE(OwnerAddress,',','.'),3) AS OwnerSplitAddress
,PARSENAME(REPLACE(OwnerAddress,',','.'),2) AS OwnerCityAddress
,PARSENAME(REPLACE(OwnerAddress,',','.'),1) AS  OwnerStateAddress
from [Nashville Housing]


ALTER TABLE [Nashville Housing]
ADD OwnerSplitAddress nvarchar(255)

Update [Nashville Housing]
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress,',','.'),3)

ALTER TABLE [Nashville Housing]
ADD OwnerCityAddress nvarchar(255)

Update [Nashville Housing]
SET OwnerCityAddress = PARSENAME(REPLACE(OwnerAddress,',','.'),2)

ALTER TABLE [Nashville Housing]
ADD OwnerStateAddress nvarchar(255)

Update [Nashville Housing]
SET OwnerStateAddress = PARSENAME(REPLACE(OwnerAddress,',','.'),1)


-- Change Y and N in "Sold as Vacant" field to  Yes and No

Select Distinct(SoldAsVacant), COUNT(SoldAsVacant) 
from [Nashville Housing]
Group by SoldAsVacant
order by 2


Select SoldAsVacant,
	CASE when SoldAsVacant = 'Y' THEN 'Yes'
		When SoldAsVacant ='N' THEN 'No'
		ELSE SoldAsVacant
		end
from [Nashville Housing]


Update [Nashville Housing]
SET SoldAsVacant = CASE when SoldAsVacant = 'Y' THEN 'Yes'
		When SoldAsVacant ='N' THEN 'No'
		ELSE SoldAsVacant
		end

-- Remove Duplicates
With RowNumCTE as(
Select *,
	ROW_NUMBER() OVER(
	PARTITION BY ParcelID,
	             PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER by
				 UniqueID
				 ) row_num
from PortfolioProject.dbo.[Nashville Housing]
--order by ParcelID
)
Select *
from RowNumCTE
where row_num>1
order by PropertyAddress

----------------------------------------------------------------------------------------
--Delete Unused Columns

Select * 
from PortfolioProject.dbo.[Nashville Housing]

Alter table PortfolioProject.dbo.[Nashville Housing]
DROP COLUMN OwnerAddress, TaxDistrict , PropertyAddress

Alter table PortfolioProject.dbo.[Nashville Housing]
DROP COLUMN SaleDate